# Real-Time Graph-Enhanced Payment Fraud Detection (BRD-DS-001, Meridian Pay)

A runnable, tested implementation of the BRD's fraud platform on **synthetic data**: streaming-style features, an entity graph
with GNN scores, anomaly detection, a calibrated ensemble, an expected-cost decision policy, reason codes, ring detection,
a scoring API with fallbacks and an audit log, and safe-rollout tooling.

> **What this is and is not.** It is a working reference build that follows the BRD's leakage, delayed-label, selection-bias and
> out-of-time rules. All data is simulated, so its metrics demonstrate correctness of the pipeline, **not** real-world fraud performance.
> Kafka, Flink, Redis, Feast and Neo4j are not integrated (an in-memory engine with the same contract stands in), and a 1,500 TPS
> load test has not been run. See `docs/ARCHITECTURE.md` for a requirement-by-requirement status and `docs/RESULTS.md` for measured results.

## Quick start
```bash
pip install -r requirements.txt
make test            # leakage, label-delay, skew, gradient-check, policy, serving, loader tests
make all             # data -> features -> train -> evaluate -> rings -> benchmark -> investigator -> report  (~20 min on 1 CPU)
make serve           # POST /v1/score on :8000
```
Individual steps: `make data features train evaluate` (≈5 min), then `make rings` (≈10 min), `make benchmark`.

Score one payment:
```bash
curl -s localhost:8000/v1/score -H 'content-type: application/json' -d '{"txn_id":1,"event_time":9000000,"is_a2a":1,"amount":2500,
 "customer_id":12,"device_id":9001,"ip_id":777,"beneficiary_id":35010,"ip_country":7,"geo_lat":9.1,"geo_lon":8.7,"ip_rep":0.7,
 "signup_time":8000000,"kyc_level":1,"risk_tier":2,"home_country":0}'
```
→ `{"decision":"review","fraud_probability":…,"probability_band":[…],"reason_codes":[{"code":"RC07",…}],"scoring_mode":"full",…}`

## Layout
```
config/            settings.yaml  policy.yaml (business-owned costs/budgets)  reason_codes.yaml
src/fraudplat/
  data/            simulator.py (ring injector, ATO, carding, delayed labels, legacy-rule selection bias)  loaders.py (IEEE-CIS, Elliptic, PaySim)
  features/        engine.py (stateful point-in-time features, shared by training and serving)  build.py
  graph/           gnn.py (NumPy GraphSAGE, cross-fit, snapshots)  rings.py (Leiden + personalised PageRank)
  models/          gbdt.py anomaly.py ensemble.py (stacker + isotonic + band) bundle.py
  decision/        policy.py       explain/ reasons.py       ops/ routing.py (shadow, canary, rollback)
  serving/         scorer.py (fallbacks, circuit breaker, audit chain)  app.py (FastAPI)
  monitoring/      reports.py (PSI/KS, fairness)       evaluation/ metrics.py
  pipelines/       data_pipeline train evaluate rings benchmark investigator report public_datasets
tests/             features (leakage), gnn (gradient check), decision_ops, serving, loaders
docs/              ARCHITECTURE.md  DATASETS.md  RESULTS.md (generated)
```

## Design choices worth knowing
* **Training/serving skew is removed by construction**: the same `FeatureEngine` is replayed in event-time order to build training data and used live. Tests compare it with an independent pandas implementation.
* **Label delay is modelled**: confirmed-fraud "taint" on devices/IPs/accounts only becomes visible after `label_time`; a test proves that changing not-yet-known labels cannot change earlier features (and that the test fails if the delay is ignored).
* **Selection bias**: payments declined by the simulated legacy rules have unobserved labels and are excluded from training except for a 3% approve-and-observe holdout.
* **GNN leakage control**: weekly snapshots, node labels only from dispositions confirmed by the training snapshot, and 2-fold cross-fitting so no account is scored by a model that saw its label.
* **Policy is config, not code**: change `config/policy.yaml` (costs, review/step-up budgets). Shadow prices are re-calibrated at train time so budgets hold.
* **Honest checks that changed the build**: results first looked too good, so the simulator was made harder; a drifting component-size feature was replaced by an edge-decayed one; a budget-calibration bug found by a test was fixed.

## Using the real public datasets
See `docs/DATASETS.md` (download steps, BRD-compliant protocols, differences from common notebooks).
