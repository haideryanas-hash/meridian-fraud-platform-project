# Results - measured on the SYNTHETIC build-phase data

> **Read this first.** All numbers below come from a simulator (`fraudplat/data/simulator.py`), not from Meridian Pay production data or the public Kaggle datasets. They prove the pipeline works end to end and are internally consistent; they are **not** a forecast of real-world fraud performance. Targets must be re-measured on real data in the Discovery phase (BRD section 15).

Model `v1` · data hash `f381cf7959c0` · config hash `ced88ae514bc` · test window = 212,015 payments, 1,543 frauds (0.73%), strictly after the training and calibration windows.

## Acceptance criteria (BRD section 16)

| # | Criterion | Threshold | Measured | Status |
|---|---|---|---|---|
| AC-1 | Fraud capture at fixed alert budget, out-of-time | 95% CI of improvement over baseline above 0 | recall@1%: 0.809 vs 0.384; paired 95% CI of difference PR-AUC [0.548, 0.599], recall@1% [0.402, 0.449] | PASS |
| AC-2 | Latency under load | p99 <= 100 ms at 1,500 TPS | p99 26.3 ms, p50 13.7 ms (single process, sequential, in-memory state); ~69 TPS per core. **1,500 TPS load test with Kafka/Flink/Redis not run** | PARTIAL |
| AC-3 | Calibration | ECE <= 0.02 | ECE 0.0011 overall (easy at 0.7% prevalence); 0.0109 within the top-5% of scores where decisions are made | PASS |
| AC-4 | Ring detection | median time-to-detect <= 3 d | median 2.96 d over 21/21 detected injected rings (p90 4.8 d; only 52% of eligible rings within 3 d); community precision 65%. Historical rings: none available | PASS (borderline) |
| AC-5 | Explainability | 100% of non-approve carry a reason code | 100% | PASS |
| AC-6 | Shadow and canary | no guardrail breach | Routing, guardrails and automatic rollback are implemented and unit-tested (`tests/test_decision_ops.py`); needs live traffic | NOT TESTED |
| AC-7 | Governance | independent validation passed | Model card and this report are inputs; Model Risk review has not happened | NOT TESTED |

## Model-level targets (BRD section 3)

| Metric | Target | Measured (final ensemble) | Baseline scorecard |
|---|---|---|---|
| PR-AUC relative lift | >= 25% | **+228%** (0.826 vs 0.251), 95% CI [0.813, 0.843] | |
| Recall at 1% review rate | >= 70% | **80.9%** (CI [0.793, 0.824]) | 38.4% |
| Dollar-weighted recall | >= 80% | **81.6%** | 52.5% |

**Alert quality (false : true alerts at matched recall, BRD objective BO-2 <= 8 : 1)**

| Recall | Precision new | False:true new | Precision scorecard | False:true scorecard |
|---|---|---|---|---|
| 40% | 99.8% | 0.00 : 1 | 25.8% | 2.87 : 1 |
| 60% | 98.6% | 0.01 : 1 | 10.0% | 8.97 : 1 |
| 80% | 63.0% | 0.59 : 1 | 3.8% | 25.34 : 1 |

At low recall the new model's highest-scored alerts are almost all fraud in this simulator (strong taint/velocity signals), so a near-zero false:true ratio there is a property of the synthetic data.

## Tier ablation (each tier must earn its latency, BRD section 12)

| Model | PR-AUC | Recall@1% | $-recall@1% |
|---|---|---|---|
| Legacy-style scorecard (baseline) | 0.251 | 0.384 | 0.525 |
| A  Tabular GBDT (velocity + behaviour) | 0.814 | 0.799 | 0.795 |
| B  + graph statistics features | 0.825 | 0.809 | 0.824 |
| C  + GNN scores (stacked) | 0.826 | 0.808 | 0.815 |
| D  + anomaly score (final ensemble) | 0.826 | 0.809 | 0.816 |

GNN as a *node classifier* on accounts that did not exist at training time (inductive): PR-AUC **0.937** vs graph-statistics baseline 0.865 (base rate 0.084). At *transaction* level the graph tiers add only a small gain here (0.814 -> 0.826) because velocity features already capture most of this simulator's signal; the graph value shows up in ring detection and the hardest typology below.

### Recall at 1% alert budget by fraud typology

| Typology | n | Final | Tabular only | Scorecard |
|---|---|---|---|---|
| ato | 319 | 0.91 | 0.90 | 0.53 |
| cnp | 557 | 0.87 | 0.88 | 0.28 |
| ring_card | 131 | 0.78 | 0.77 | 0.25 |
| ring_cashout | 205 | 0.99 | 1.00 | 0.61 |
| ring_in | 331 | 0.50 | 0.46 | 0.33 |

## Decision policy simulation (same review budget; realised-cost model in `config/policy.yaml`)

| | Legacy-style (scorecard + rules) | New (expected-cost policy) |
|---|---|---|
| Fraud loss, bps of TPV | 124.1 | 20.2 |
| Decline rate | 0.19% | 0.75% |
| Review rate | 0.47% | 0.54% |
| Step-up rate | 0.00% | 1.91% |
| False : true alerts | 2.10 | 1.20 |

Simulated fraud-loss reduction: 84%. **Do not compare this with the BRD's -30% target**: the simulated legacy baseline is a deliberately simple scorecard, assumed step-up/review effectiveness (85% / 95%) drives much of the gap, and the simulator is not calibrated to Meridian's 11.5 bps. The meaningful, like-for-like comparison is recall at a fixed alert budget (AC-1).

## Monitoring

* **Drift** (PSI vs the latest mature window, alert > 0.2): features over threshold per period {'day130-160': 1, 'day160-190': 1, 'day190-210': 4}; score PSI day100-130 0.002, day130-160 0.003, day160-190 0.004, day190-210 0.011.
* **Fairness screen** (friction on legitimate payments by `region` and `home_country`, alert if > 1.25x overall): 0 flagged segments of 15. The simulator has no built-in segment bias, so a pass here is not evidence about real populations.

## Live path

* Online vs offline parity over 6,000 payments: max |feature diff| = 0.0e+00, max |p diff| = 0.0e+00, decision agreement 100.0%.
* Audit log: 6,000 records, hash chain valid = True, replay reproduces 100.0% of decisions.
* Fallback `full_down`: modes {'light': 400}, no unscored approval above the risk limit = True.
* Fallback `full_and_light_down`: modes {'rules': 400}, no unscored approval above the risk limit = True.
* HTTP (in-process ASGI): p50 25.8 ms, p99 38.9 ms.

## Ring detection

Flag threshold chosen on the validation window only (11.0). Test window:

| Seeds | Rings eligible | Detected | Median TTD (d) | P90 TTD (d) | Detected <= 3 d | Community precision | Flagged / day |
|---|---|---|---|---|---|---|---|
| model flags + confirmed labels | 21 | 21 | 2.96 | 4.84 | 52% | 65% | 12.9 |
| confirmed labels only | 21 | 21 | 4.82 | 7.79 | 19% | 95% | 6.4 |

The BRD's 19-day baseline is not reproduced by the simulator, so no improvement factor is claimed.
