"""Benchmarks on the public proxy datasets with the BRD protocol (temporal splits, no leakage, no SMOTE).

    python -m fraudplat.pipelines.public_datasets ieee    --dir data/raw/ieee_cis
    python -m fraudplat.pipelines.public_datasets elliptic --dir data/raw/elliptic
    python -m fraudplat.pipelines.public_datasets paysim  --dir data/raw/paysim --max-rows 1000000
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, precision_recall_fscore_support, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from ..config import artifacts_dir
from ..data.loaders import ieee_features, load_elliptic, load_ieee_cis, load_paysim, paysim_to_events, time_split
from ..evaluation import metrics as M
from ..features.build import replay
from ..features.engine import FEATURE_NAMES
from ..graph.gnn import HeteroSAGE, Snapshot
from ..models.gbdt import margin, sigmoid, train_gbdt


def _save(name, res):
    d = artifacts_dir() / "reports"
    d.mkdir(exist_ok=True)
    (d / f"public_{name}.json").write_text(json.dumps(res, indent=2, default=float))
    return res


# ------------------------------------------------------------------ IEEE-CIS
def run_ieee(path, nrows=None, n_rounds=600, verbose=True) -> dict:
    df = load_ieee_cis(path, nrows)
    y = df["isFraud"].to_numpy()
    tr, va, te = time_split(df["TransactionDT"].to_numpy())
    X = ieee_features(df, tr)
    cols = list(X.columns)
    gb = train_gbdt(X[tr].to_numpy(np.float32), y[tr], X[va].to_numpy(np.float32), y[va], cols, n_rounds=n_rounds, monotone=False)
    s_te = sigmoid(margin(gb, X[te].to_numpy(np.float32)))
    s_va = sigmoid(margin(gb, X[va].to_numpy(np.float32)))
    iso = IsotonicRegression(out_of_bounds="clip", y_min=0, y_max=1).fit(s_va, y[va])
    base_cols = [c for c in ("amt_log", "hour", "dow", "card1", "addr1", "C1", "D1") if c in cols]
    base = make_pipeline(StandardScaler(), LogisticRegression(class_weight="balanced", max_iter=300))
    Xb = X[base_cols].fillna(-1).to_numpy(np.float32)
    base.fit(Xb[tr], y[tr])
    s_base = base.predict_proba(Xb[te])[:, 1]
    amt = df["TransactionAmt"].to_numpy()[te]
    res = {"dataset": "IEEE-CIS", "rows": len(df), "prevalence": float(y.mean()), "split": "time-ordered 60/20/20",
           "test": M.summary(y[te], s_te, amt), "baseline_logistic": M.summary(y[te], s_base, amt),
           "ece_calibrated": M.ece(y[te], iso.predict(s_te)), "best_iteration": gb.best_iteration,
           "pr_auc_ci95": M.bootstrap_ci(y[te], s_te, lambda a, b: average_precision_score(a, b), 100)}
    res["pr_auc_relative_lift"] = res["test"]["pr_auc"] / res["baseline_logistic"]["pr_auc"] - 1
    if verbose:
        print(json.dumps(res, indent=1, default=float))
    return _save("ieee_cis", res)


# ------------------------------------------------------------------ Elliptic
def _norm_adj(n, rows, cols):
    deg = np.bincount(rows, minlength=n).astype(np.float32)
    return sp.csr_matrix((1.0 / np.maximum(deg[rows], 1), (rows, cols)), shape=(n, n), dtype=np.float32)


def _best_f1_threshold(y, s):
    best, thr = -1, 0.5
    for t in np.unique(np.quantile(s, np.linspace(0.5, 0.999, 120))):
        f = precision_recall_fscore_support(y, s >= t, average="binary", zero_division=0)[2]
        if f > best:
            best, thr = f, t
    return float(thr)


def _illicit_report(y, s, thr):
    p, r, f, _ = precision_recall_fscore_support(y, s >= thr, average="binary", zero_division=0)
    return {"pr_auc": float(average_precision_score(y, s)), "roc_auc": float(roc_auc_score(y, s)),
            "illicit_precision": float(p), "illicit_recall": float(r), "illicit_f1": float(f), "threshold_from_validation": thr}


def run_elliptic(path, epochs=120, hidden=32, verbose=True) -> dict:
    d = load_elliptic(path)
    X, y, ts, n = d["X"], d["y"], d["time_step"], len(d["y"])
    train_n, val_n, test_n = (ts <= 29), (ts >= 30) & (ts <= 34), (ts >= 35)
    assert ts[train_n].max() < ts[val_n].min() <= ts[val_n].max() < ts[test_n].min()          # strictly out-of-time
    lab = y >= 0
    mu, sd = X[train_n].mean(0), X[train_n].std(0) + 1e-6          # scaler fit on the training period only
    Xs = np.clip((X - mu) / sd, -5, 5).astype(np.float32)
    # directed edges -> two relations: "who pays me" (in) and "whom I pay" (out); mean aggregation
    A_in, A_out = _norm_adj(n, d["dst"], d["src"]), _norm_adj(n, d["src"], d["dst"])
    nbr = (A_in @ Xs + A_out @ Xs) / 2
    res = {"dataset": "Elliptic", "nodes": n, "labelled": int(lab.sum()), "illicit_share_labelled": float(y[lab].mean()),
           "split": "train steps 1-29, validation 30-34, test 35-49"}
    idx_tr, idx_va, idx_te = (np.flatnonzero(m & lab) for m in (train_n, val_n, test_n))

    def gbdt_scores(F, name):
        g = train_gbdt(F[idx_tr], y[idx_tr], F[idx_va], y[idx_va], [f"{name}{i}" for i in range(F.shape[1])], 400, monotone=False)
        sv, st = sigmoid(margin(g, F[idx_va])), sigmoid(margin(g, F[idx_te]))
        return _illicit_report(y[idx_te], st, _best_f1_threshold(y[idx_va], sv))

    res["lightgbm_node_features_only"] = gbdt_scores(Xs, "f")
    res["lightgbm_plus_neighbour_means"] = gbdt_scores(np.hstack([Xs, nbr]), "g")
    snap = Snapshot()
    snap.X, snap.A = Xs, [A_in, A_out]
    t0 = time.time()
    m1 = HeteroSAGE(hidden, 0, f_dim=Xs.shape[1], n_rel=2)
    # training uses only train-period labels; message passing never crosses time steps in Elliptic
    m1.fit(snap, np.where(lab, y, 0).astype(np.float32), idx_tr, epochs=epochs, lr=0.01, wd=1e-4)
    lg = m1.predict_logits(snap, n)
    res["graphsage_inductive"] = _illicit_report(y[idx_te], sigmoid(lg[idx_te]), _best_f1_threshold(y[idx_va], sigmoid(lg[idx_va])))
    res["graphsage_train_seconds"] = round(time.time() - t0, 1)
    if verbose:
        print(json.dumps(res, indent=1, default=float))
    return _save("elliptic", res)


# ------------------------------------------------------------------ PaySim
def run_paysim(path, max_rows=1_000_000, verbose=True) -> dict:
    df = load_paysim(path, nrows=None if not max_rows else max_rows)
    ev = paysim_to_events(df)
    t0 = time.time()
    X, _ = replay(ev, verbose=False)
    dt = time.time() - t0
    y = ev["is_fraud"].to_numpy()
    tr, va, te = time_split(ev["event_time"].to_numpy())
    # PaySim fraud exists only in TRANSFER / CASH_OUT; features never touch balance columns (they were dropped)
    gb = train_gbdt(X[tr], y[tr], X[va], y[va], FEATURE_NAMES, 400, monotone=False)
    s = sigmoid(margin(gb, X[te]))
    res = {"dataset": "PaySim", "rows": len(ev), "prevalence": float(y.mean()), "split": "time-ordered 60/20/20",
           "balance_columns_used": False, "engine_replay_seconds": round(dt, 1), "engine_events_per_second": round(len(ev) / dt),
           "test": M.summary(y[te], s, ev["amount"].to_numpy()[te])}
    if verbose:
        print(json.dumps(res, indent=1, default=float))
    return _save("paysim", res)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dataset", choices=["ieee", "elliptic", "paysim"])
    ap.add_argument("--dir", required=True)
    ap.add_argument("--max-rows", type=int, default=1_000_000)
    a = ap.parse_args()
    {"ieee": lambda: run_ieee(a.dir, a.max_rows if a.max_rows else None), "elliptic": lambda: run_elliptic(a.dir),
     "paysim": lambda: run_paysim(a.dir, a.max_rows)}[a.dataset]()


if __name__ == "__main__":
    main()
