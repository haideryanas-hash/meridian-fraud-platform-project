"""Live-path benchmark: online/offline parity, latency, fallbacks, audit replay (NFR-01, NFR-04, NFR-07, FR-03, FR-12)."""
from __future__ import annotations

import json
import time

import numpy as np
import pandas as pd

from ..config import DAY, EVENT_COLS, artifacts_dir, load_settings
from ..features.engine import FEATURE_NAMES
from ..serving.app import PaymentEvent, build_scorer, create_app
from ..serving.scorer import AuditLog, replay_record
from .common import load_events


def _events(ev: pd.DataFrame, lo_day: int, n: int):
    sub = ev[ev.event_time >= lo_day * DAY].iloc[:n]
    return sub[EVENT_COLS + ["is_fraud", "label_time", "label_observed", "taint_dev", "taint_sender"]].to_dict("records")


def run(n_events: int = 6000, http_n: int = 400, verbose: bool = True) -> dict:
    S = load_settings()
    day0 = S["splits"]["test_start_day"]
    ev = load_events()
    audit = artifacts_dir() / "reports" / "audit_benchmark.jsonl"
    audit.parent.mkdir(exist_ok=True)
    t0 = time.time()
    scorer = build_scorer(warm_until_day=day0, audit_path=audit)
    warm_s = time.time() - t0
    events = _events(ev, day0, n_events)
    outs = []
    for e in events:                                           # feed events and delayed labels exactly like production would
        outs.append(scorer.score(e))
        if e["is_fraud"] and e["label_observed"]:
            scorer.engine.schedule_label(e)
    lat = scorer.latency_summary()
    assert lat["modes"] == {"full": len(events)} and not lat["tier_errors"], f"healthy path degraded: {lat['modes']} {lat['last_errors']}"

    # ---- parity with the offline pipeline (training/serving skew)
    ids = [e["txn_id"] for e in events]
    off_f = pd.read_parquet(artifacts_dir().parent / "data" / "features.parquet").set_index("txn_id").loc[ids, FEATURE_NAMES].to_numpy(np.float32)
    on_f = np.array([json.loads(l)["record"]["features"] for l in open(audit)], np.float32)
    off_p = pd.read_parquet(artifacts_dir() / "decisions.parquet").set_index("txn_id").loc[ids]
    on_p = np.array([o["fraud_probability"] for o in outs])
    on_dec = np.array([o["decision"] for o in outs])
    parity = {"feature_max_abs_diff": float(np.abs(off_f - on_f).max()), "p_cal_max_abs_diff": float(np.abs(off_p.p_cal.to_numpy() - on_p).max()),
              "decision_agreement": float((off_p.action_name.to_numpy() == on_dec).mean())}

    # ---- audit chain + replay
    ok, n = AuditLog.verify(audit)
    rows = [json.loads(l)["record"] for l in open(audit)][:1500]
    from ..models.bundle import ModelBundle
    bundle = scorer.b
    same = np.mean([replay_record(r, bundle)["action"] == r["action"] for r in rows])
    cod = np.mean([len(o["reason_codes"]) >= 1 for o in outs if o["decision"] != "approve"])

    # ---- fallbacks (BR-07): inject tier outages
    fb = {}
    for name, faults in (("full_down", {"full"}), ("full_and_light_down", {"full", "light"})):
        sc = build_scorer(warm_until_day=day0)
        sc.faults = faults
        res = [sc.score(e, log_audit=False) for e in _events(ev, day0, 400)]
        big = [r for e, r in zip(_events(ev, day0, 400), res) if e["amount"] >= sc.policy.risk_limit]
        fb[name] = {"modes": sc.latency_summary()["modes"], "no_unscored_approval_above_limit": all(r["decision"] != "approve" for r in big if r["scoring_mode"] == "rules"),
                    "p99_ms": sc.latency_summary()["p99_ms"]}

    # ---- HTTP path (in-process ASGI client, includes pydantic + JSON overhead, excludes network)
    from fastapi.testclient import TestClient
    sc = build_scorer(warm_until_day=day0)
    client = TestClient(create_app(sc))
    http_lat = []
    for e in _events(ev, day0, http_n):
        body = PaymentEvent(**{k: e[k] for k in PaymentEvent.model_fields}).model_dump()
        t1 = time.perf_counter()
        r = client.post("/v1/score", json=body)
        http_lat.append((time.perf_counter() - t1) * 1000)
        assert r.status_code == 200
    out = {"events_scored": len(events), "warm_up_replay_seconds": round(warm_s, 1), "latency": lat,
           "single_core_throughput_tps": 1000.0 / np.mean(scorer.latencies), "parity_vs_offline": parity,
           "audit": {"records": n, "chain_valid": ok, "replay_reproduces_decision": float(same)},
           "reason_code_coverage_non_approve": float(cod), "fallbacks": fb,
           "http": {"n": len(http_lat), "p50_ms": float(np.percentile(http_lat, 50)), "p99_ms": float(np.percentile(http_lat, 99))},
           "decision_mix": pd.Series(on_dec).value_counts(normalize=True).round(4).to_dict()}
    (artifacts_dir() / "reports" / "benchmark.json").write_text(json.dumps(out, indent=2, default=float))
    if verbose:
        print(json.dumps({k: out[k] for k in ("latency", "single_core_throughput_tps", "parity_vs_offline", "audit", "reason_code_coverage_non_approve", "http")}, indent=1, default=float))
        print("fallbacks:", json.dumps(fb))
    return out


if __name__ == "__main__":
    run()
