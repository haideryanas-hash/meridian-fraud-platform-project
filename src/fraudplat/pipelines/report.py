"""Assemble docs/RESULTS.md (acceptance-criteria scorecard) and the model card from the measured JSON reports.
Every number in these documents is read from a pipeline output; nothing is typed in by hand."""
from __future__ import annotations

import json

from ..config import ROOT, artifacts_dir

PASS, FAIL, PART, NA = "PASS", "FAIL", "PARTIAL", "NOT TESTED"


def _j(name):
    p = artifacts_dir() / "reports" / name
    return json.loads(p.read_text()) if p.exists() else None


def build() -> str:
    m, r, b = _j("metrics.json"), _j("rings.json"), _j("benchmark.json")
    meta = json.loads((artifacts_dir() / "models" / m["model_version"] / "runtime.json").read_text())["meta"]
    ab, fin, base = m["ablation"], m["ablation"]["p_stack"], m["ablation"]["scorecard"]
    pol, cal = m["policy_simulation"], m["calibration"]
    L = []
    L.append("# Results - measured on the SYNTHETIC build-phase data\n")
    L.append("> **Read this first.** All numbers below come from a simulator (`fraudplat/data/simulator.py`), not from Meridian Pay production data "
             "or the public Kaggle datasets. They prove the pipeline works end to end and are internally consistent; they are **not** a forecast of real-world "
             "fraud performance. Targets must be re-measured on real data in the Discovery phase (BRD section 15).\n")
    L.append(f"Model `{m['model_version']}` · data hash `{meta['data_hash']}` · config hash `{meta['config_hash']}` · "
             f"test window = {m['test_rows']:,} payments, {m['test_frauds']:,} frauds ({m['test_prevalence']:.2%}), strictly after the training and calibration windows.\n")

    # --- acceptance criteria
    ci = m["paired_diff_vs_scorecard_ci95"]
    ac1 = PASS if ci["pr_auc"][0] > 0 and ci["recall_at_1pct"][0] > 0 else FAIL
    lat = b["latency"] if b else None
    ac2 = (PART if lat and lat["p99_ms"] <= 100 else FAIL) if lat else NA
    t = r["test_with_model_seeds"] if r else None
    ac4 = ((PASS + (" (borderline)" if t["median_ttd_days_detected"] > 2.7 else "")) if t["median_ttd_days_detected"] is not None and t["median_ttd_days_detected"] <= 3 else FAIL) if t else NA
    ac5 = (PASS if b and b["reason_code_coverage_non_approve"] == 1.0 else FAIL) if b else NA
    L.append("## Acceptance criteria (BRD section 16)\n")
    L.append("| # | Criterion | Threshold | Measured | Status |\n|---|---|---|---|---|")
    L.append(f"| AC-1 | Fraud capture at fixed alert budget, out-of-time | 95% CI of improvement over baseline above 0 | recall@1%: {fin['recall_at_1pct']:.3f} vs {base['recall_at_1pct']:.3f}; "
             f"paired 95% CI of difference PR-AUC [{ci['pr_auc'][0]:.3f}, {ci['pr_auc'][1]:.3f}], recall@1% [{ci['recall_at_1pct'][0]:.3f}, {ci['recall_at_1pct'][1]:.3f}] | {ac1} |")
    if lat:
        L.append(f"| AC-2 | Latency under load | p99 <= 100 ms at 1,500 TPS | p99 {lat['p99_ms']:.1f} ms, p50 {lat['p50_ms']:.1f} ms (single process, sequential, in-memory state); "
                 f"~{b['single_core_throughput_tps']:.0f} TPS per core. **1,500 TPS load test with Kafka/Flink/Redis not run** | {ac2} |")
    L.append(f"| AC-3 | Calibration | ECE <= 0.02 | ECE {cal['ece']:.4f} overall (easy at {m['test_prevalence']:.1%} prevalence); {cal['ece_top5pct_scores']:.4f} within the top-5% of scores where decisions are made | {PASS if cal['ece'] <= 0.02 and cal['ece_top5pct_scores'] <= 0.02 else FAIL} |")
    if t:
        L.append(f"| AC-4 | Ring detection | median time-to-detect <= 3 d | median {t['median_ttd_days_detected']:.2f} d over {t['rings_detected']}/{t['rings_eligible']} detected injected rings "
                 f"(p90 {t['p90_ttd_days_detected']:.1f} d; only {t['share_detected_within_3d']:.0%} of eligible rings within 3 d); community precision {t['community_precision']:.0%}. Historical rings: none available | {ac4} |")
    if b:
        L.append(f"| AC-5 | Explainability | 100% of non-approve carry a reason code | {b['reason_code_coverage_non_approve']:.0%} | {ac5} |")
    L.append(f"| AC-6 | Shadow and canary | no guardrail breach | Routing, guardrails and automatic rollback are implemented and unit-tested (`tests/test_decision_ops.py`); needs live traffic | {NA} |")
    L.append(f"| AC-7 | Governance | independent validation passed | Model card and this report are inputs; Model Risk review has not happened | {NA} |\n")

    # --- BRD targets
    L.append("## Model-level targets (BRD section 3)\n")
    L.append("| Metric | Target | Measured (final ensemble) | Baseline scorecard |\n|---|---|---|---|")
    L.append(f"| PR-AUC relative lift | >= 25% | **{m['pr_auc_relative_lift_vs_scorecard']:+.0%}** ({fin['pr_auc']:.3f} vs {base['pr_auc']:.3f}), 95% CI [{fin['pr_auc_ci95'][0]:.3f}, {fin['pr_auc_ci95'][1]:.3f}] | |")
    L.append(f"| Recall at 1% review rate | >= 70% | **{fin['recall_at_1pct']:.1%}** (CI [{fin['recall_at_1pct_ci95'][0]:.3f}, {fin['recall_at_1pct_ci95'][1]:.3f}]) | {base['recall_at_1pct']:.1%} |")
    L.append(f"| Dollar-weighted recall | >= 80% | **{fin['dollar_recall_at_1pct']:.1%}** | {base['dollar_recall_at_1pct']:.1%} |")
    L.append("\n**Alert quality (false : true alerts at matched recall, BRD objective BO-2 <= 8 : 1)**\n")
    L.append("| Recall | Precision new | False:true new | Precision scorecard | False:true scorecard |\n|---|---|---|---|---|")
    for c in m["alert_quality"]["curve"]:
        L.append(f"| {c['recall']:.0%} | {c['precision_new']:.1%} | {c['false_to_true_new']:.2f} : 1 | {c['precision_scorecard']:.1%} | {c['false_to_true_scorecard']:.2f} : 1 |")
    L.append("\nAt low recall the new model's highest-scored alerts are almost all fraud in this simulator (strong taint/velocity signals), so a near-zero false:true ratio there is a property of the synthetic data.\n")

    L.append("## Tier ablation (each tier must earn its latency, BRD section 12)\n")
    L.append("| Model | PR-AUC | Recall@1% | $-recall@1% |\n|---|---|---|---|")
    for k, v in ab.items():
        L.append(f"| {v['label']} | {v['pr_auc']:.3f} | {v['recall_at_1pct']:.3f} | {v['dollar_recall_at_1pct']:.3f} |")
    g = meta["gnn"]
    L.append(f"\nGNN as a *node classifier* on accounts that did not exist at training time (inductive): PR-AUC **{g['pr_auc_gnn_inductive']:.3f}** vs graph-statistics baseline "
             f"{g['pr_auc_graph_stats_baseline']:.3f} (base rate {g['base_rate']:.3f}). At *transaction* level the graph tiers add only a small gain here "
             f"({ab['p_light']['pr_auc']:.3f} -> {fin['pr_auc']:.3f}) because velocity features already capture most of this simulator's signal; "
             f"the graph value shows up in ring detection and the hardest typology below.\n")
    L.append("### Recall at 1% alert budget by fraud typology\n")
    L.append("| Typology | n | Final | Tabular only | Scorecard |\n|---|---|---|---|---|")
    for k, v in m["recall_at_1pct_by_fraud_type"].items():
        L.append(f"| {k} | {v['n']} | {v['recall_new']:.2f} | {v['recall_tabular_only']:.2f} | {v['recall_scorecard']:.2f} |")

    L.append("\n## Decision policy simulation (same review budget; realised-cost model in `config/policy.yaml`)\n")
    L.append("| | Legacy-style (scorecard + rules) | New (expected-cost policy) |\n|---|---|---|")
    for lab, key, fmt in [("Fraud loss, bps of TPV", "fraud_loss_bps_of_tpv", "{:.1f}"), ("Decline rate", "decline_rate", "{:.2%}"), ("Review rate", "review_rate", "{:.2%}"),
                          ("Step-up rate", "step_up_rate", "{:.2%}"), ("False : true alerts", "false_to_true_alerts", "{:.2f}")]:
        L.append(f"| {lab} | {fmt.format(pol['legacy_style'][key])} | {fmt.format(pol['new_policy'][key])} |")
    L.append(f"\nSimulated fraud-loss reduction: {pol['fraud_loss_reduction']:.0%}. **Do not compare this with the BRD's -30% target**: the simulated legacy baseline is a deliberately simple scorecard, "
             f"assumed step-up/review effectiveness (85% / 95%) drives much of the gap, and the simulator is not calibrated to Meridian's 11.5 bps. "
             f"The meaningful, like-for-like comparison is recall at a fixed alert budget (AC-1).\n")

    d, f = m["drift"], m["fairness"]
    L.append("## Monitoring\n")
    L.append(f"* **Drift** (PSI vs the latest mature window, alert > {d['psi_alert_threshold']}): features over threshold per period {d['features_over_threshold'] or 'none'}; score PSI "
             f"{', '.join(f'{x['period']} {x['psi']:.3f}' for x in d['score_drift'])}.")
    L.append(f"* **Fairness screen** (friction on legitimate payments by `region` and `home_country`, alert if > {1.25}x overall): {len(f['flags'])} flagged segments of {len(f['segments'])}. "
             f"The simulator has no built-in segment bias, so a pass here is not evidence about real populations.\n")
    if b:
        L.append("## Live path\n")
        p = b["parity_vs_offline"]
        L.append(f"* Online vs offline parity over {b['events_scored']:,} payments: max |feature diff| = {p['feature_max_abs_diff']:.1e}, max |p diff| = {p['p_cal_max_abs_diff']:.1e}, decision agreement {p['decision_agreement']:.1%}.")
        L.append(f"* Audit log: {b['audit']['records']:,} records, hash chain valid = {b['audit']['chain_valid']}, replay reproduces {b['audit']['replay_reproduces_decision']:.1%} of decisions.")
        for k, v in b["fallbacks"].items():
            L.append(f"* Fallback `{k}`: modes {v['modes']}, no unscored approval above the risk limit = {v['no_unscored_approval_above_limit']}.")
        L.append(f"* HTTP (in-process ASGI): p50 {b['http']['p50_ms']:.1f} ms, p99 {b['http']['p99_ms']:.1f} ms.\n")
    if r:
        L.append("## Ring detection\n")
        L.append(f"Flag threshold chosen on the validation window only ({r['chosen_flag_score']}). Test window:\n")
        L.append("| Seeds | Rings eligible | Detected | Median TTD (d) | P90 TTD (d) | Detected <= 3 d | Community precision | Flagged / day |\n|---|---|---|---|---|---|---|---|")
        for lab, k in (("model flags + confirmed labels", "test_with_model_seeds"), ("confirmed labels only", "test_labels_only_seeds")):
            x = r[k]
            L.append(f"| {lab} | {x['rings_eligible']} | {x['rings_detected']} | {x['median_ttd_days_detected']:.2f} | {x['p90_ttd_days_detected']:.2f} | {x['share_detected_within_3d']:.0%} | {x['community_precision']:.0%} | {x['avg_flagged_per_day']:.1f} |")
        L.append("\nThe BRD's 19-day baseline is not reproduced by the simulator, so no improvement factor is claimed.\n")

    txt = "\n".join(L)
    (ROOT / "docs").mkdir(exist_ok=True)
    (ROOT / "docs" / "RESULTS.md").write_text(txt)

    card = f"""# Model card - fraud scoring ensemble {m['model_version']}
**Intended use:** score card-not-present and account-to-account payments before authorisation; output a calibrated fraud probability with uncertainty band, a policy decision and reason codes.
**Not for:** AML / sanctions, card-present channels, automated adverse action without human-reviewable reason codes.

| Item | Value |
|---|---|
| Tiers | LightGBM (tabular + graph statistics) -> stacked with GNN node scores and Isolation Forest anomaly -> isotonic calibration; tabular LightGBM as fallback |
| Training window | days 0-{meta['splits']['train_end_day']} ({meta['n_train_rows']:,} rows, {meta['n_train_pos']} frauds) with delayed + observed labels only; stacker fit days {meta['splits']['train_end_day']}-{meta['splits']['stack_split_day']}; calibration days {meta['splits']['stack_split_day']}-{meta['splits']['val_end_day']} |
| Evaluation | out-of-time days >= {meta['splits']['test_start_day']} |
| Data hash / config hash | `{meta['data_hash']}` / `{meta['config_hash']}` |
| Libraries | {meta['libs']} |
| Excluded from features | geography proxies (`home_country`, `region`), raw IDs, any label-derived field newer than its label_time |
| Known limitations | trained on synthetic data; labels for legacy-declined payments are unobserved (reject-inference is excluded, not corrected); graph snapshots refresh weekly; edges are time-stamped but union-free (edge decay 30 d) |
| Monitoring | PSI / KS per feature and score, fairness screen, label-matured performance - see `monitoring/reports.py` |
| Top features (gain) | {', '.join(list(meta['top_features_gain'])[:8])} |
"""
    (artifacts_dir() / "models" / m["model_version"] / "model_card.md").write_text(card)
    return txt


if __name__ == "__main__":
    print(build())
