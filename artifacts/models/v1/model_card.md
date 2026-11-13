# Model card - fraud scoring ensemble v1
**Intended use:** score card-not-present and account-to-account payments before authorisation; output a calibrated fraud probability with uncertainty band, a policy decision and reason codes.
**Not for:** AML / sanctions, card-present channels, automated adverse action without human-reviewable reason codes.

| Item | Value |
|---|---|
| Tiers | LightGBM (tabular + graph statistics) -> stacked with GNN node scores and Isolation Forest anomaly -> isotonic calibration; tabular LightGBM as fallback |
| Training window | days 0-70 (107,929 rows, 563 frauds) with delayed + observed labels only; stacker fit days 70-88; calibration days 88-100 |
| Evaluation | out-of-time days >= 100 |
| Data hash / config hash | `f381cf7959c0` / `ced88ae514bc` |
| Libraries | {'python': '3.13.9', 'lightgbm': '4.7.0', 'sklearn': '1.9.1', 'numpy': '2.5.4'} |
| Excluded from features | geography proxies (`home_country`, `region`), raw IDs, any label-derived field newer than its label_time |
| Known limitations | trained on synthetic data; labels for legacy-declined payments are unobserved (reject-inference is excluded, not corrected); graph snapshots refresh weekly; edges are time-stamped but union-free (edge decay 30 d) |
| Monitoring | PSI / KS per feature and score, fairness screen, label-matured performance - see `monitoring/reports.py` |
| Top features (gain) | ip_rep, amount, ip_cnt_30d, geo_kmh, account_age_days, hour, ip_sum_30d, mer_cnt_7d |
