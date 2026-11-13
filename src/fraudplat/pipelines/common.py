from __future__ import annotations

import hashlib

import numpy as np
import pandas as pd

from ..config import DAY, artifacts_dir, data_dir

BASELINE_FEATURES = ["log_amount", "is_a2a", "ip_country_mismatch", "ip_rep", "kyc_level", "risk_tier",
                     "account_age_days", "cust_prior_txns", "amount_z"]


def load_events() -> pd.DataFrame:
    return pd.read_parquet(data_dir() / "events.parquet")


def load_features(names: list[str]) -> np.ndarray:
    return pd.read_parquet(data_dir() / "features.parquet")[names].to_numpy(np.float32)


def scorecard_matrix(X: np.ndarray, names: list[str]) -> np.ndarray:
    """Design matrix of the legacy-style scorecard: a handful of static / profile variables only."""
    cols = [X[:, names.index(c)] for c in BASELINE_FEATURES]
    hour = X[:, names.index("hour")]
    cols.append((hour < 6).astype(float))
    cols[BASELINE_FEATURES.index("account_age_days")] = np.log1p(cols[BASELINE_FEATURES.index("account_age_days")])
    cols[BASELINE_FEATURES.index("cust_prior_txns")] = np.log1p(cols[BASELINE_FEATURES.index("cust_prior_txns")])
    return np.column_stack(cols)


def data_hash(ev: pd.DataFrame) -> str:
    h = pd.util.hash_pandas_object(ev[["txn_id", "event_time", "amount", "is_fraud"]], index=False).to_numpy()
    return hashlib.sha256(h.tobytes()).hexdigest()[:12]


def days(ev: pd.DataFrame) -> np.ndarray:
    return ev["event_time"].to_numpy() / DAY
