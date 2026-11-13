# Architecture and BRD traceability

```
payment event ──► FeatureEngine (stateful, point-in-time) ──► GBDT full ─┐
  (is_a2a, device,   velocity 1m..30d x 6 entities            GNN scores ─┼► Stacker (LR) ► Isotonic calibrator ► p + 95% band
   ip, amount ...)   behaviour, geo, graph + taint (delayed)  Isolation F.─┘                        │
                                                                                                   ▼
        delayed labels ◄── /v1/labels ◄── case mgmt        DecisionPolicy (expected cost, config) ► approve / step_up / review / decline
                                                                       │                                   │
                                                  hash-chained audit log (features, versions)      TreeSHAP ► reason codes
   offline: same engine replayed in event-time order ► feature table ► training ► weekly GNN snapshots ► ring detection (Leiden + PPR)
```

## Requirement -> implementation
| BRD id | Implemented in | Status |
|---|---|---|
| FR-01 idempotent ingest | `FeatureEngine.process` is deterministic over ordered events; no Kafka consumer | **Simulated** (interface only) |
| FR-02 velocity features, 6 windows x 6 entities | `features/engine.py` | Done |
| FR-03 online/offline identical definitions | single engine for both; `tests/test_features.py::test_online_matches_independent_offline_reference`; end-to-end parity in `benchmark.py` | Done (in-memory store, not Redis/Feast) |
| FR-04 point-in-time training sets, leakage tests | replay design + future-truncation, label-delay (with negative control) and no-label-access tests | Done |
| FR-05 graph freshness, neighbour/community features | per-event graph state (zero lag), 2-hop taint, edge decay | Done (in-memory, not Neo4j/TigerGraph) |
| FR-06 inductive GNN | `graph/gnn.py` NumPy GraphSAGE, cross-fitted, weekly snapshots; gradient-checked | Done (no PyTorch Geometric) |
| FR-07 calibrated probability + uncertainty band | `models/ensemble.py` (isotonic + Wilson band per plateau) | Done |
| FR-08 expected-cost policy, segment thresholds, alert budget | `decision/policy.py`, `config/policy.yaml` | Done |
| FR-09 TreeSHAP reason codes | LightGBM `pred_contrib` (TreeSHAP) mapped by `config/reason_codes.yaml` | Done (stacker terms attributed linearly) |
| FR-10 ring detection | `graph/rings.py`, `pipelines/rings.py` | Done (batch daily, not streaming) |
| FR-11 investigator view | `pipelines/investigator.py` static HTML | Done (no similar-case *retrieval index*, only nearest cases in the run) |
| FR-12 immutable log + replay | `serving/scorer.py::AuditLog`, `replay_record` | Done (hash chain in a file, not WORM storage) |
| FR-13 shadow / canary / rollback | `ops/routing.py` | Done as library + tests; not wired to a live router |
| FR-14 monitoring | `monitoring/reports.py` PSI / KS / fairness | Batch reports; no dashboards or paging |
| FR-15 governed retraining | deterministic `train` pipeline, versioned bundle | Manual trigger; no Airflow / MLflow |
| BR-07 / NFR-04 degrade, never unscored approve | scorer fallback chain + circuit breaker | Done, tested |
| NFR-01 latency | benchmark reports measured p50/p99 | Measured single-process |
| NFR-02 1,500 TPS, NFR-03 99.95% active-active | - | **Not built or tested** |
| NFR-05/06 PCI, GDPR | - | **Not applicable to a synthetic build** |
| NFR-07 7-year auditability | audit record format supports replay | Format only |
| NFR-08 reproducibility | fixed seeds, deterministic LightGBM, retrain gives identical predictions | Done |

## Deliberate departures from the BRD technology list
* **Streaming / stores** (Kafka, Flink, Feast, Redis, Neo4j): replaced by an in-memory engine with the same contract so that logic can be tested offline.
  A production port keeps `FeatureEngine.compute/update` semantics and swaps the state containers.
* **GNN**: pure NumPy/SciPy because PyTorch could not be installed in the build sandbox. The architecture (relation-specific mean aggregation,
  fixed fan-out sampling, inductive weights) maps one-to-one onto PyTorch Geometric `SAGEConv`/`HeteroConv` if you prefer it.
* **Anomaly tier**: Isolation Forest only (BRD allows an autoencoder as an alternative).
