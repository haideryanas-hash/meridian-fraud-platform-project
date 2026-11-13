"""Training pipeline: strict out-of-time splits, matured/observed labels only (BRD 9.4, 12)."""
from __future__ import annotations

import json
import platform
import time

import lightgbm
import numpy as np
import pandas as pd
import sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from ..config import DAY, artifacts_dir, config_hash, load_settings, load_yaml
from ..decision.policy import DecisionPolicy
from ..features.engine import FEATURE_NAMES, TABULAR_FEATURES
from ..graph.gnn import build_gnn_store
from ..models.anomaly import AnomalyTier
from ..models.bundle import ModelBundle
from ..models.ensemble import STACK_COLS, Calibrator, Stacker, stack_inputs
from ..models.gbdt import margin, sigmoid, train_gbdt
from .common import data_hash, days, load_events, load_features, scorecard_matrix


def run(version: str = "v1", verbose: bool = True):
    S = load_settings()
    sp = S["splits"]
    ev = load_events()
    X = load_features(FEATURE_NAMES)
    y = ev["is_fraud"].to_numpy()
    d = days(ev)
    obs = ev["label_observed"].to_numpy() == 1
    amount, a2a = ev["amount"].to_numpy(), ev["is_a2a"].to_numpy()
    tab_idx = [FEATURE_NAMES.index(n) for n in TABULAR_FEATURES]
    log = print if verbose else (lambda *a, **k: None)

    tr = (d < sp["train_end_day"] - sp["early_stop_days"]) & obs
    es = (d >= sp["train_end_day"] - sp["early_stop_days"]) & (d < sp["train_end_day"]) & obs
    train_all = d < sp["train_end_day"]
    val_a = (d >= sp["train_end_day"]) & (d < sp["stack_split_day"]) & obs
    val_b = (d >= sp["stack_split_day"]) & (d < sp["val_end_day"]) & obs
    post = d >= sp["train_end_day"]
    log(f"rows train={tr.sum()} (pos {y[tr].sum()}) es={es.sum()} (pos {y[es].sum()}) valA={val_a.sum()} (pos {y[val_a].sum()}) "
        f"valB={val_b.sum()} (pos {y[val_b].sum()})")

    # ---- baseline scorecard (stand-in for the legacy logistic-regression scorecard)
    t0 = time.time()
    sc = make_pipeline(StandardScaler(), LogisticRegression(class_weight="balanced", max_iter=500))
    sc.fit(scorecard_matrix(X[tr | es], FEATURE_NAMES), y[tr | es])
    # ---- Tier 1 GBDT: tabular-only (fallback + ablation) and full (tabular + graph statistics)
    g_light = train_gbdt(X[tr][:, tab_idx], y[tr], X[es][:, tab_idx], y[es], TABULAR_FEATURES, monotone=False)
    g_full = train_gbdt(X[tr], y[tr], X[es], y[es], FEATURE_NAMES)
    log(f"GBDT light best_iter={g_light.best_iteration}  full best_iter={g_full.best_iteration}  ({time.time() - t0:.0f}s)")

    # ---- Tier 3 anomaly (unsupervised, whole training window)
    an = AnomalyTier(FEATURE_NAMES, seed=0).fit(X[train_all])

    # ---- Tier 2 GNN (weekly snapshots, cross-fitted)
    t0 = time.time()
    store, gnn_report = build_gnn_store(ev, S["gnn"], sp["train_end_day"] * DAY, S["simulation"]["n_days"], verbose=verbose)
    log(f"GNN inductive PR-AUC {gnn_report['pr_auc_gnn_inductive']:.3f} vs graph-stats baseline "
        f"{gnn_report['pr_auc_graph_stats_baseline']:.3f}  ({time.time() - t0:.0f}s)")

    # ---- features for all post-train rows
    P = np.flatnonzero(post)
    Xp = X[P]
    m_full = margin(g_full, Xp)
    an_p = an.score(Xp)
    gs, gb, gd = store.lookup(ev["event_time"].to_numpy()[P], ev["customer_id"].to_numpy()[P],
                              ev["beneficiary_id"].to_numpy()[P], ev["device_id"].to_numpy()[P])
    Z_full = stack_inputs(m_full, an_p, gs, gb, gd)
    sel = lambda mask: mask[P]  # noqa: E731

    # ---- Tier 4: stacker (time-split: fit on valA) and isotonic calibration on the most recent mature window valB
    cols_gnn = ["gbdt_margin", "gnn_send", "gnn_ben", "gnn_dev"]
    st_gnn = Stacker(cols_gnn).fit(stack_inputs(m_full, an_p, gs, gb, gd, cols_gnn)[sel(val_a)], y[P][sel(val_a)])
    st = Stacker(STACK_COLS).fit(Z_full[sel(val_a)], y[P][sel(val_a)])
    raw_logit = st.logit(Z_full)
    cal = Calibrator().fit(sigmoid(raw_logit)[sel(val_b)], y[P][sel(val_b)])
    p_cal, lo, hi = cal.predict_with_band(sigmoid(raw_logit))
    cal_light = Calibrator().fit(sigmoid(margin(g_light, Xp[:, tab_idx]))[sel(val_b)], y[P][sel(val_b)])   # fallback tier
    log("stacker coefficients:", {k: round(v, 3) for k, v in st.coef.items()})

    # ---- policy: calibrate the review shadow price so the manual-review budget holds on valB
    pol = DecisionPolicy(load_yaml(S["policy_file"]))
    shadow, shadow_su = pol.calibrate_shadow_price(p_cal[sel(val_b)], amount[P][sel(val_b)], a2a[P][sel(val_b)])
    log(f"shadow prices: review={shadow:.2f} (budget {pol.budget:.2%}), step-up={shadow_su:.2f} (budget {pol.su_budget:.2%})")

    # ---- predictions table for evaluation
    preds = pd.DataFrame({
        "txn_id": ev["txn_id"].to_numpy()[P],
        "scorecard": sc.predict_proba(scorecard_matrix(Xp, FEATURE_NAMES))[:, 1],
        "p_light": sigmoid(margin(g_light, Xp[:, tab_idx])),
        "p_full": sigmoid(m_full),
        "p_stack_gnn": sigmoid(st_gnn.logit(stack_inputs(m_full, an_p, gs, gb, gd, cols_gnn))),
        "p_stack": sigmoid(raw_logit), "p_cal": p_cal, "band_lo": lo, "band_hi": hi,
        "anomaly": an_p, "gnn_send": gs, "gnn_ben": gb, "gnn_dev": gd})
    preds.to_parquet(artifacts_dir() / "predictions.parquet")

    imp = pd.Series(g_full.feature_importance("gain"), index=FEATURE_NAMES).sort_values(ascending=False)
    meta = {"model_version": version, "trained_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "data_hash": data_hash(ev), "config_hash": config_hash(S, load_yaml(S["policy_file"])),
            "libs": {"python": platform.python_version(), "lightgbm": lightgbm.__version__, "sklearn": sklearn.__version__,
                     "numpy": np.__version__},
            "splits": sp, "gbdt_best_iteration": {"full": g_full.best_iteration, "light": g_light.best_iteration},
            "stacker_coef": st.coef, "gnn": gnn_report, "top_features_gain": imp.head(15).round(1).to_dict(),
            "n_train_rows": int(tr.sum()), "n_train_pos": int(y[tr].sum())}
    bundle = ModelBundle(FEATURE_NAMES, g_full, g_light, an, st, cal, store, shadow, meta, shadow_su, cal_light)
    out = artifacts_dir() / "models" / version
    bundle.save(out)
    import joblib
    joblib.dump(sc, out / "scorecard.joblib")
    log("saved bundle ->", out)
    return bundle


if __name__ == "__main__":
    run()
