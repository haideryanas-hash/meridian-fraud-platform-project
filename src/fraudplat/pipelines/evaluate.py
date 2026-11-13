"""Out-of-time evaluation: tier ablation, baseline comparison with CIs, calibration, policy simulation, monitoring."""
from __future__ import annotations

import json

import matplotlib
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, precision_recall_curve

from ..config import DAY, artifacts_dir, load_settings, load_yaml
from ..decision.policy import ACTIONS, DecisionPolicy
from ..evaluation import metrics as M
from ..features.engine import FEATURE_NAMES
from ..models.bundle import ModelBundle
from ..monitoring.reports import drift_report, fairness_report
from .common import load_events, load_features

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

TIERS = [("scorecard", "Legacy-style scorecard (baseline)"), ("p_light", "A  Tabular GBDT (velocity + behaviour)"),
         ("p_full", "B  + graph statistics features"), ("p_stack_gnn", "C  + GNN scores (stacked)"),
         ("p_stack", "D  + anomaly score (final ensemble)")]


def _band_coverage(test: pd.DataFrame, min_n: int = 200) -> dict:
    """Do the Wilson bands fitted on the calibration window contain the fraud rate observed out-of-time?"""
    g = test.assign(pk=test["p_cal"].round(10)).groupby("pk").agg(n=("is_fraud", "size"), obs=("is_fraud", "mean"),
                                                                 lo=("band_lo", "first"), hi=("band_hi", "first"))
    g = g[g.n >= min_n]
    ok = (g.obs >= g.lo) & (g.obs <= g.hi)
    return {"score_bins_checked": int(len(g)), "coverage_weighted": float(np.average(ok, weights=g.n)),
            "mean_band_width": float(np.average(g.hi - g.lo, weights=g.n))}


def run(version: str = "v1", n_boot: int = 100, verbose: bool = True):
    S = load_settings()
    sp = S["splits"]
    rep_dir = artifacts_dir() / "reports"
    rep_dir.mkdir(exist_ok=True)
    ev = load_events()
    pr = pd.read_parquet(artifacts_dir() / "predictions.parquet")
    df = pr.merge(ev, on="txn_id")
    bundle = ModelBundle.load(artifacts_dir() / "models" / version)
    test = df[df.event_time >= sp["test_start_day"] * DAY].reset_index(drop=True)
    y, amt, a2a = test["is_fraud"].to_numpy(), test["amount"].to_numpy(), test["is_a2a"].to_numpy()
    out: dict = {"model_version": version, "test_rows": int(len(test)), "test_frauds": int(y.sum()),
                 "test_prevalence": float(y.mean())}

    # ---------------- tier ablation
    abl = {}
    for col, label in TIERS:
        s = test[col].to_numpy()
        r = M.summary(y, s, amt)
        r["label"] = label
        abl[col] = r
    for col in ("scorecard", "p_stack"):
        abl[col]["pr_auc_ci95"] = M.bootstrap_ci(y, test[col].to_numpy(), lambda yy, ss: average_precision_score(yy, ss), n_boot)
        abl[col]["recall_at_1pct_ci95"] = M.bootstrap_ci(y, test[col].to_numpy(), lambda yy, ss: M.recall_at_rate(yy, ss, 0.01), n_boot)
    out["ablation"] = abl
    fin, base = test["p_stack"].to_numpy(), test["scorecard"].to_numpy()
    out["paired_diff_vs_scorecard_ci95"] = M.paired_bootstrap(y, fin, base, amt, n_boot=n_boot)
    out["pr_auc_relative_lift_vs_scorecard"] = abl["p_stack"]["pr_auc"] / abl["scorecard"]["pr_auc"] - 1

    # ---------------- BO-2: alert precision at matched recall
    base_recall = abl["scorecard"]["recall_at_1pct"]
    p_new = M.precision_at_recall(y, fin, base_recall)
    p_old = abl["scorecard"]["precision_at_1pct"]
    out["alert_quality"] = {"matched_recall": base_recall, "precision_baseline": p_old, "precision_new": p_new,
                            "false_to_true_baseline": (1 - p_old) / p_old, "false_to_true_new": (1 - p_new) / p_new}
    curve = []
    for tr_ in (0.4, 0.6, 0.8):
        pn, pb = M.precision_at_recall(y, fin, tr_), M.precision_at_recall(y, base, tr_)
        curve.append({"recall": tr_, "precision_new": pn, "precision_scorecard": pb,
                      "false_to_true_new": (1 - pn) / pn, "false_to_true_scorecard": (1 - pb) / pb})
    out["alert_quality"]["curve"] = curve

    # ---------------- recall by fraud typology at the 1% alert budget
    hit = M.top_rate_mask(fin, 0.01)
    byt = {}
    mb = M.top_rate_mask(base, 0.01)
    for ft, g in test[test.is_fraud == 1].groupby("fraud_type"):
        byt[ft] = {"n": int(len(g)), "recall_new": float(hit[g.index].mean()), "recall_scorecard": float(mb[g.index].mean()),
                   "recall_tabular_only": float(M.top_rate_mask(test["p_light"].to_numpy(), 0.01)[g.index].mean())}
    out["recall_at_1pct_by_fraud_type"] = byt

    # ---------------- calibration
    pc = test["p_cal"].to_numpy()
    top = pc >= np.quantile(pc, 0.95)
    out["calibration"] = {"ece": M.ece(y, pc), "ece_top5pct_scores": M.ece(y[top], pc[top], bins=5),
                          "brier": M.brier(y, pc), "brier_prior_only": M.brier(y, np.full(len(y), y.mean())),
                          "mean_pred": float(pc.mean()), "prevalence": float(y.mean()),
                          "reliability": M.reliability_table(y, pc, 10),
                          "uncertainty_band": _band_coverage(test)}

    # ---------------- policy simulation vs legacy-style decisions (same alert budget)
    pol = DecisionPolicy(load_yaml(S["policy_file"]), shadow_price=bundle.shadow_price, step_up_shadow=bundle.step_up_shadow)
    act_new = pol.decide(pc, amt, a2a)
    n_review = int((act_new == 2).sum())
    thr = np.sort(base)[::-1][max(n_review - 1, 0)]
    act_old = np.where(test["legacy_declined"].to_numpy() == 1, 3, np.where(base >= thr, 2, 0))
    tpv = float(amt.sum())
    sim = {}
    for name, act in (("legacy_style", act_old), ("new_policy", act_new)):
        oc = pol.outcome_costs(act, y, amt, a2a)
        alerts = act >= 2
        sim[name] = {**oc, "fraud_loss_bps_of_tpv": oc["fraud_loss"] / tpv * 1e4,
                     "decline_rate": float((act == 3).mean()), "review_rate": float((act == 2).mean()),
                     "step_up_rate": float((act == 1).mean()),
                     "alert_precision": float(y[alerts].mean()) if alerts.any() else None,
                     "false_to_true_alerts": float((1 - y[alerts].mean()) / max(y[alerts].mean(), 1e-9)) if alerts.any() else None,
                     "fraud_amount_not_stopped_pct": float((amt * y)[act == 0].sum() / (amt * y).sum())}
    sim["fraud_loss_reduction"] = 1 - sim["new_policy"]["fraud_loss"] / sim["legacy_style"]["fraud_loss"]
    sim["total_cost_reduction"] = 1 - sim["new_policy"]["total_cost"] / sim["legacy_style"]["total_cost"]
    out["policy_simulation"] = sim
    out["policy_shadow_price"] = {"review": bundle.shadow_price, "step_up": bundle.step_up_shadow}

    # ---------------- cost frontier: fraud loss vs review budget (Pareto of loss vs friction)
    fr = []
    for bud in (0.001, 0.0025, 0.005, 0.01, 0.02):
        p2 = DecisionPolicy(dict(pol.cfg, review_rate_budget=bud))
        sh, _ = p2.calibrate_shadow_price(pc, amt, a2a)
        a_ = p2.decide(pc, amt, a2a)
        oc = p2.outcome_costs(a_, y, amt, a2a)
        fr.append({"review_budget": bud, "shadow_price": sh, "review_rate": float((a_ == 2).mean()),
                   "decline_rate": float((a_ == 3).mean()), "fraud_loss": oc["fraud_loss"], "total_cost": oc["total_cost"]})
    out["cost_frontier"] = fr

    # ---------------- monitoring: drift + fairness (BR-10)
    X_all = load_features(FEATURE_NAMES)
    dd = ev["event_time"].to_numpy() / DAY
    ref = X_all[(dd >= sp["train_end_day"]) & (dd < sp["test_start_day"])]     # latest mature window
    periods = {f"day{a}-{b}": X_all[(dd >= a) & (dd < b)] for a, b in [(100, 130), (130, 160), (160, 190), (190, 210)]}
    score_ref = df.loc[(df.event_time < sp["test_start_day"] * DAY), "p_stack"].to_numpy()     # val window scores
    score_cur = {k: test.loc[(test.event_time >= a * DAY) & (test.event_time < b * DAY), "p_stack"].to_numpy()
                 for k, (a, b) in {"day100-130": (100, 130), "day130-160": (130, 160), "day160-190": (160, 190), "day190-210": (190, 210)}.items()}
    out["drift"] = drift_report(ref[::4], {k: v[::4] for k, v in periods.items()}, FEATURE_NAMES, score_ref, score_cur,
                                S["monitoring"]["psi_alert"])
    fdf = test.assign(action=act_new)
    out["fairness"] = fairness_report(fdf, ["region", "home_country"], S["fairness"]["ratio_alert"], S["fairness"]["min_segment_n"])

    # ---------------- persist decisions (used by ring detection) + charts
    dcols = ["txn_id", "event_time", "customer_id", "beneficiary_id", "device_id", "ip_id", "amount", "is_a2a", "p_cal",
             "band_lo", "band_hi", "is_fraud", "fraud_type", "campaign_id", "label_time", "label_observed", "taint_dev",
             "taint_sender", "signup_time"]
    dec = df[dcols].copy()                                          # all post-train rows (val rows are in-sample)
    dec["action"] = pol.decide(df["p_cal"].to_numpy(), df["amount"].to_numpy(), df["is_a2a"].to_numpy())
    dec["action_name"] = [ACTIONS[i] for i in dec["action"]]
    dec.to_parquet(artifacts_dir() / "decisions.parquet")

    fig, ax = plt.subplots(1, 3, figsize=(16, 4.4))
    for col, label in TIERS:
        pr_, rc_, _ = precision_recall_curve(y, test[col].to_numpy())
        ax[0].plot(rc_, pr_, label=label.split("(")[0].strip(), lw=1.6)
    ax[0].set(xlabel="Recall", ylabel="Precision", title="Out-of-time PR curves (test window)")
    ax[0].legend(fontsize=7)
    rel = out["calibration"]["reliability"]
    ax[1].loglog([r["mean_pred"] for r in rel], [max(r["obs_rate"], 1e-5) for r in rel], "o-")
    ax[1].plot([1e-5, 1], [1e-5, 1], "k--", lw=.8)
    ax[1].set(xlabel="Mean predicted probability", ylabel="Observed fraud rate", title="Reliability (10 equal-mass bins)")
    ax[2].plot([f["review_rate"] * 100 for f in fr], [f["fraud_loss"] / 1e3 for f in fr], "o-")
    ax[2].set(xlabel="Manual review rate (%)", ylabel="Fraud loss (k USD, test)", title="Loss vs alert budget")
    plt.tight_layout()
    plt.savefig(rep_dir / "evaluation.png", dpi=130)
    plt.close()
    (rep_dir / "metrics.json").write_text(json.dumps(out, indent=2, default=float))
    if verbose:
        a = abl
        print("tier ablation (test PR-AUC / recall@1%):")
        for col, label in TIERS:
            print(f"  {label:45s} {a[col]['pr_auc']:.3f} / {a[col]['recall_at_1pct']:.3f}")
        print("policy:", {k: round(v, 3) for k, v in sim["new_policy"].items() if k in ("fraud_loss_bps_of_tpv", "decline_rate", "review_rate", "step_up_rate", "false_to_true_alerts")})
        print("legacy:", {k: round(v, 3) for k, v in sim["legacy_style"].items() if k in ("fraud_loss_bps_of_tpv", "decline_rate", "review_rate", "false_to_true_alerts")})
        print("fraud loss reduction %.1f%%   ECE %.5f" % (100 * sim["fraud_loss_reduction"], out["calibration"]["ece"]))
        print("fairness flags:", out["fairness"]["flags"][:4], "| drift features>PSI:", out["drift"]["features_over_threshold"])
    return out


if __name__ == "__main__":
    run()
