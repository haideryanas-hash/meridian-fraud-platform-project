"""Tier 3: unsupervised anomaly detection (Isolation Forest). Labels are never used; the output is a
percentile against the training distribution, used as a secondary score in the stacker (BRD 9.3)."""
from __future__ import annotations

import numpy as np
from sklearn.ensemble import IsolationForest

ANOMALY_FEATURES = ["log_amount", "amount_z", "amount_over_max", "secs_since_last", "cust_burst_1h", "geo_dist_km",
                    "geo_kmh", "cust_cnt_10m", "cust_cnt_1h", "cust_cnt_24h", "dev_cnt_1h", "dev_cnt_24h",
                    "ip_cnt_1h", "ip_cnt_24h", "card_cnt_1h", "new_dev_for_cust", "new_mer_for_cust",
                    "new_ben_for_cust", "hour", "is_a2a", "ip_rep", "ip_country_mismatch", "account_age_days"]


class AnomalyTier:
    def __init__(self, feature_names, n_estimators=60, seed=0):
        self.cols = [feature_names.index(c) for c in ANOMALY_FEATURES]
        self.model = IsolationForest(n_estimators=n_estimators, max_samples=2048, random_state=seed, n_jobs=1)
        self.ref = None

    def fit(self, X):
        Z = X[:, self.cols]
        self.model.fit(Z)
        self.ref = np.sort(-self.model.score_samples(Z))
        return self

    def score(self, X) -> np.ndarray:
        """Percentile in [0,1] of the raw anomaly score within the training distribution."""
        Z = np.atleast_2d(X)[:, self.cols]
        raw = -self.model.score_samples(Z)
        return np.searchsorted(self.ref, raw) / len(self.ref)
