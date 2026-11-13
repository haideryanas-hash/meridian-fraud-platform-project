"""Ring-detection backtest on injected rings: precision and time-to-detect (BO-4, AC-4)."""
from __future__ import annotations

from pathlib import Path
import argparse
import gc
import json
import pickle
import os

try:
    import resource
except ImportError:
    resource = None
import time

import numpy as np
import pandas as pd

def get_memory_usage():
    if resource is not None:
        return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024

    try:
        import psutil
        process = psutil.Process(os.getpid())
        return process.memory_info().rss / (1024 * 1024)
    except ImportError:
        return None

from ..config import DAY, artifacts_dir, load_settings
from ..graph.rings import OFF_D, OFF_I, detect
from .common import load_events


def ground_truth(ev: pd.DataFrame):
    f = ev[ev.fraud_type.str.startswith("ring")]
    rings = {}
    for cid, g in f.groupby("campaign_id"):
        mules = np.unique(np.concatenate([g.loc[g.fraud_type == "ring_in", "beneficiary_id"].to_numpy(),
                                          g.loc[g.fraud_type == "ring_cashout", "customer_id"].to_numpy()]))
        rings[int(cid)] = {"first_fraud": int(g.event_time.min()), "mules": mules, "n_fraud": int(len(g))}
    camp = {}
    for cid, g in ev[ev.is_fraud == 1].groupby("campaign_id"):
        camp[int(cid)] = np.unique(np.concatenate([g.customer_id.to_numpy(), g.beneficiary_id.to_numpy()[g.beneficiary_id.to_numpy() >= 0]]))
    return rings, camp


def _label_map(groups: dict, n: int):
    m = np.full(n, -1)
    for cid, accts in groups.items():
        m[accts[accts < n]] = cid
    return m


def backtest(dec: pd.DataFrame, ev: pd.DataFrame, cfg: dict, start_day: int, end_day: int, thresholds, use_model_seeds=True,
             eval_margin_days: int = 5, edge_threshold: float | None = None, verbose=False):
    """One detection pass per day, evaluated at every flag threshold. Returns ({thr: result}, cases_at_detection)."""
    rings, camp = ground_truth(ev)
    n_acct = int(max(ev.customer_id.max(), ev.beneficiary_id.max())) + 1
    mule_map = _label_map({k: v["mules"] for k, v in rings.items()}, n_acct)
    camp_map = _label_map(camp, n_acct)
    signup = np.full(n_acct, np.nan)
    first = ev.drop_duplicates("customer_id")
    signup[first.customer_id.to_numpy()] = first.signup_time.to_numpy()
    dec = dec.sort_values("event_time").reset_index(drop=True)
    t = dec.event_time.to_numpy()
    fr = dec[(dec.is_fraud == 1) & (dec.label_observed == 1)].sort_values("label_time").reset_index(drop=True)
    lt = fr.label_time.to_numpy()
    thresholds = sorted(thresholds)
    st = {x: {"detected": {}, "flagged": 0, "tp": 0, "daily": []} for x in thresholds}
    cases_keep = {}
    t0 = time.time()
    for d in range(start_day + 1, end_day + 1):
        now = d * DAY
        lo = max(start_day * DAY, now - cfg["window_days"] * DAY)
        win = dec.iloc[np.searchsorted(t, lo): np.searchsorted(t, now)]
        conf = fr.iloc[: np.searchsorted(lt, now, side="right")]
        b = conf.beneficiary_id.to_numpy()
        seed_acct = [b[b >= 0], conf.customer_id.to_numpy()[conf.taint_sender.to_numpy() == 1]]
        td = conf.taint_dev.to_numpy() == 1
        seed_infra = [OFF_D + conf.device_id.to_numpy()[td], OFF_I + conf.ip_id.to_numpy()[td]]
        if use_model_seeds:
            fl = win[win.action >= 2]
            fb = fl.beneficiary_id.to_numpy()
            seed_acct.append(fb[fb >= 0])
            seed_infra += [OFF_D + fl.device_id.to_numpy(), OFF_I + fl.ip_id.to_numpy()]
        cases = detect(win, np.concatenate(seed_acct), np.concatenate(seed_infra), now, signup, cfg,
                       return_edges=edge_threshold is not None, min_score=thresholds[0])
        for c in cases:
            acc = np.array(c["accounts"])
            acc = acc[acc < n_acct]
            rid, cid = mule_map[acc], camp_map[acc]
            hit_ring = [int(r) for r, n in zip(*np.unique(rid[rid >= 0], return_counts=True)) if n >= 3]
            hit_camp = [int(r) for r, n in zip(*np.unique(cid[cid >= 0], return_counts=True)) if n >= 3]
            for x in thresholds:
                if c["score"] < x:
                    continue
                st[x]["flagged"] += 1
                st[x]["tp"] += int(bool(hit_ring or hit_camp))
                for r in hit_ring:
                    st[x]["detected"].setdefault(r, d)
                    if edge_threshold == x and r not in cases_keep:
                        cases_keep[r] = dict(c, ring_id=r, detected_day=d)
        for x in thresholds:
            st[x]["daily"].append(sum(1 for c in cases if c["score"] >= x))
        gc.collect()
        if verbose and d % 10 == 0:
            rss = get_memory_usage()
            print(f"  day {d} ({time.time() - t0:.0f}s, peak RSS {rss:.0f} MB)", flush=True)

    eligible = {r: v for r, v in rings.items() if start_day * DAY <= v["first_fraud"] <= (end_day - eval_margin_days) * DAY}
    out = {}
    for x in thresholds:
        detected = st[x]["detected"]
        ttd = {r: (detected[r] * DAY - v["first_fraud"]) / DAY for r, v in eligible.items() if r in detected}
        vals = np.array(list(ttd.values()), float)
        out[x] = {"window_days": [start_day, end_day], "flag_score": x, "rings_eligible": len(eligible), "rings_detected": len(ttd),
                  "detection_rate": len(ttd) / max(len(eligible), 1),
                  "median_ttd_days_detected": float(np.median(vals)) if len(vals) else None,
                  "p90_ttd_days_detected": float(np.percentile(vals, 90)) if len(vals) else None,
                  "share_detected_within_3d": float((vals <= 3).sum() / max(len(eligible), 1)),
                  "flagged_communities_total": st[x]["flagged"], "community_precision": st[x]["tp"] / max(st[x]["flagged"], 1),
                  "avg_flagged_per_day": float(np.mean(st[x]["daily"])), "use_model_seeds": use_model_seeds,
                  "ttd_by_ring": {str(k): v for k, v in ttd.items()}}
    return out, cases_keep


KEYS = ("rings_eligible", "rings_detected", "median_ttd_days_detected", "share_detected_within_3d", "community_precision", "avg_flagged_per_day")
GRID = [3.0, 5.0, 7.0, 9.0, 11.0]


def _rep() -> "Path":
    d = artifacts_dir() / "reports"
    d.mkdir(exist_ok=True)
    return d


def stage_dev(verbose=True):
    """Choose the flag threshold on the validation window ONLY (never on the test window)."""
    S = load_settings()
    sp = S["splits"]
    dev, _ = backtest(pd.read_parquet(artifacts_dir() / "decisions.parquet"), load_events(), dict(S["rings"]), sp["train_end_day"],
                      sp["test_start_day"], GRID)
    best = max(GRID, key=lambda x: dev[x]["share_detected_within_3d"] * dev[x]["community_precision"])
    out = {"grid": {str(k): {x: dev[k][x] for x in KEYS} for k in GRID}, "chosen_flag_score": best}
    (_rep() / "rings_dev.json").write_text(json.dumps(out, indent=2, default=float))
    if verbose:
        for k, v in out["grid"].items():
            print("dev flag_score", k, {a: (round(b, 3) if isinstance(b, float) else b) for a, b in v.items()})
        print("chosen flag_score:", best)
    return out


def stage_test(model_seeds: bool, verbose=True):
    S = load_settings()
    sp = S["splits"]
    best = json.loads((_rep() / "rings_dev.json").read_text())["chosen_flag_score"]
    res, cases = backtest(pd.read_parquet(artifacts_dir() / "decisions.parquet"), load_events(), dict(S["rings"]), sp["test_start_day"],
                          S["simulation"]["n_days"], [best], model_seeds, edge_threshold=best if model_seeds else None, verbose=verbose)
    name = "with_model_seeds" if model_seeds else "labels_only_seeds"
    (_rep() / f"rings_test_{name}.json").write_text(json.dumps(res[best], indent=2, default=float))
    if model_seeds:
        with open(artifacts_dir() / "ring_cases.pkl", "wb") as f:
            pickle.dump(cases, f)
    if verbose:
        print(name, {x: (round(v, 3) if isinstance(v, float) else v) for x, v in res[best].items() if x != "ttd_by_ring"})


def merge():
    d = _rep()
    dev = json.loads((d / "rings_dev.json").read_text())
    out = {"dev_threshold_search": dev["grid"], "chosen_flag_score": dev["chosen_flag_score"],
           "test_with_model_seeds": json.loads((d / "rings_test_with_model_seeds.json").read_text()),
           "test_labels_only_seeds": json.loads((d / "rings_test_labels_only_seeds.json").read_text())}
    (d / "rings.json").write_text(json.dumps(out, indent=2))
    return out


def run(verbose: bool = True):
    """All stages in one process (needs more RAM); `make rings` runs each stage in its own process."""
    stage_dev(verbose)
    stage_test(True, verbose)
    stage_test(False, verbose)
    return merge()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["all", "dev", "test", "test_labels", "merge"], default="all")
    a = ap.parse_args()
    {"all": run, "dev": stage_dev, "test": lambda: stage_test(True), "test_labels": lambda: stage_test(False), "merge": merge}[a.stage]()
