"""Offline feature generation by replaying events through the SAME engine used for live scoring (FR-03/FR-04)."""
from __future__ import annotations

import time

import numpy as np
import pandas as pd

from ..config import EVENT_COLS
from .engine import FEATURE_NAMES, FeatureEngine

_LABEL_FIELDS = ["is_fraud", "label_time", "label_observed", "taint_dev", "taint_sender"]


def replay(df: pd.DataFrame, engine: FeatureEngine | None = None, scan_cap: int = 400,
           verbose: bool = True, chunk: int = 20000):
    """Process events in event-time order. Returns (X float32 [n, n_features], engine)."""
    assert df["event_time"].is_monotonic_increasing, "events must be in event-time order"
    engine = engine or FeatureEngine(scan_cap)
    cols = EVENT_COLS + _LABEL_FIELDS
    X = np.empty((len(df), len(FEATURE_NAMES)), np.float32)
    t0 = time.time()
    for s in range(0, len(df), chunk):
        for i, ev in enumerate(df.iloc[s:s + chunk][cols].to_dict("records")):
            X[s + i] = engine.process(ev)
            if ev["is_fraud"] and ev["label_observed"]:       # confirmed-fraud feedback, visible only after label_time
                engine.schedule_label(ev)
        if verbose:
            print(f"  replayed {min(s + chunk, len(df)):>7}/{len(df)}  ({time.time() - t0:.0f}s)", flush=True)
    return X, engine


def feature_frame(df: pd.DataFrame, X: np.ndarray) -> pd.DataFrame:
    out = pd.DataFrame(X, columns=FEATURE_NAMES)
    out.insert(0, "txn_id", df["txn_id"].to_numpy())
    return out
