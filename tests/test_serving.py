"""End-to-end serving tests against the trained bundle (skipped if artifacts are missing)."""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from fraudplat.config import DAY, EVENT_COLS, ROOT

BUNDLE = ROOT / "artifacts" / "models" / "v1"
pytestmark = pytest.mark.skipif(not (BUNDLE / "runtime.json").exists() or not (ROOT / "artifacts" / "decisions.parquet").exists(),
                                reason="run the training pipeline first")


@pytest.fixture(scope="module")
def world(tmp_path_factory):
    from fraudplat.serving.app import build_scorer
    ev = pd.read_parquet(ROOT / "data" / "events.parquet")
    audit = tmp_path_factory.mktemp("audit") / "audit.jsonl"
    scorer = build_scorer(warm_until_day=100, audit_path=audit)
    rows = ev[ev.event_time >= 100 * DAY].iloc[:400]
    evs = rows[EVENT_COLS + ["is_fraud", "label_time", "label_observed", "taint_dev", "taint_sender"]].to_dict("records")
    outs = []
    for e in evs:
        outs.append(scorer.score(e))
        if e["is_fraud"] and e["label_observed"]:
            scorer.engine.schedule_label(e)
    return scorer, evs, outs, audit, ev


def test_healthy_path_uses_full_ensemble_and_hides_no_errors(world):
    """Regression: a TypeError in the stacker was silently absorbed by the fallback chain (everything scored in 'light')."""
    scorer, _, outs, _, _ = world
    assert {o["scoring_mode"] for o in outs} == {"full"}
    assert scorer.errors == {} and scorer.latency_summary()["tier_errors"] == {}


def test_real_tier_errors_are_recorded_not_swallowed(world):
    from fraudplat.serving.app import build_scorer
    _, evs, _, _, _ = world
    sc = build_scorer(warm_until_day=100)
    sc._full = lambda x, ev: (_ for _ in ()).throw(ValueError("boom"))          # a genuine bug, not an injected outage
    out = sc.score(evs[0], log_audit=False)
    assert out["scoring_mode"] == "light" and sc.errors == {"full": 1} and "boom" in sc.last_errors[0]


def test_online_matches_offline_pipeline(world):
    scorer, evs, outs, _, _ = world
    dec = pd.read_parquet(ROOT / "artifacts" / "decisions.parquet").set_index("txn_id")
    ids = [e["txn_id"] for e in evs]
    off = dec.loc[ids]
    on = np.array([o["fraud_probability"] for o in outs])
    assert np.abs(off.p_cal.to_numpy() - on).max() < 1e-6
    assert (off.action_name.to_numpy() == np.array([o["decision"] for o in outs])).mean() == 1.0


def test_reason_codes_on_every_non_approve(world):
    _, _, outs, _, _ = world
    non = [o for o in outs if o["decision"] != "approve"]
    assert non and all(len(o["reason_codes"]) >= 1 for o in non)
    assert all(o["probability_band"][0] <= o["fraud_probability"] <= o["probability_band"][1] + 1e-9 for o in outs)


def test_latency_budget_single_request(world):
    scorer = world[0]
    s = scorer.latency_summary()
    assert s["p99_ms"] < 250        # generous CI bound; the benchmark reports the real number vs the 100 ms SLO


def test_audit_chain_and_replay(world):
    from fraudplat.serving.scorer import AuditLog, replay_record
    scorer, _, _, audit, _ = world
    ok, n = AuditLog.verify(audit)
    assert ok and n == 400
    recs = [json.loads(l)["record"] for l in open(audit)]
    assert all(replay_record(r, scorer.b)["action"] == r["action"] for r in recs)
    lines = open(audit).read().splitlines()
    tampered = audit.with_name("t.jsonl")
    row = json.loads(lines[10]); row["record"]["action"] = "approve" if row["record"]["action"] != "approve" else "decline"
    lines[10] = json.dumps(row, sort_keys=True)
    tampered.write_text("\n".join(lines) + "\n")
    assert AuditLog.verify(tampered)[0] is False


def test_fallback_chain_never_unscored_approval_above_limit(world):
    from fraudplat.serving.app import build_scorer
    _, evs, _, _, _ = world
    sc = build_scorer(warm_until_day=100)
    sc.faults = {"full"}
    assert {sc.score(e, log_audit=False)["scoring_mode"] for e in evs[:50]} == {"light"}
    sc.faults = {"full", "light"}
    risky = dict(evs[60], amount=sc.policy.risk_limit + 1, event_time=evs[60]["event_time"])
    out = sc.score(risky, log_audit=False)
    assert out["scoring_mode"] == "rules" and out["decision"] != "approve"


def test_http_endpoints(world):
    from fastapi.testclient import TestClient
    from fraudplat.serving.app import PaymentEvent, build_scorer, create_app
    _, evs, _, _, _ = world
    client = TestClient(create_app(build_scorer(warm_until_day=100)))
    assert client.get("/healthz").json()["status"] == "ok"
    body = PaymentEvent(**{k: evs[0][k] for k in PaymentEvent.model_fields}).model_dump()
    r = client.post("/v1/score", json=body)
    assert r.status_code == 200 and r.json()["decision"] in ("approve", "step_up", "review", "decline")
    assert client.post("/v1/score", json={"txn_id": 1}).status_code == 422          # schema validation
    assert client.get("/v1/metrics").json()["n"] == 1
