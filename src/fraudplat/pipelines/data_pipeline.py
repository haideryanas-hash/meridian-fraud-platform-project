"""Step 1-2: generate the synthetic proxy data and the point-in-time feature table."""
from __future__ import annotations

import time

import pandas as pd

from ..config import data_dir, load_settings
from ..data.simulator import simulate
from ..features.build import feature_frame, replay
from .common import data_hash


def generate(verbose=True) -> pd.DataFrame:
    S = load_settings()
    ev = simulate(S["simulation"], S["labels"])
    ev.to_parquet(data_dir() / "events.parquet")
    if verbose:
        f = ev[ev.is_fraud == 1]
        print(f"events={len(ev):,}  fraud={len(f):,} ({ev.is_fraud.mean():.3%})  data_hash={data_hash(ev)}")
        print("fraud types:", f.fraud_type.value_counts().to_dict())
    return ev


def build_features(verbose=True):
    S = load_settings()
    ev = pd.read_parquet(data_dir() / "events.parquet")
    t0 = time.time()
    X, _ = replay(ev, scan_cap=S["features"]["scan_cap"], verbose=False)
    feature_frame(ev, X).to_parquet(data_dir() / "features.parquet")
    if verbose:
        print(f"features {X.shape} in {time.time() - t0:.0f}s ({(time.time() - t0) / len(ev) * 1e6:.0f} us/event)")


if __name__ == "__main__":
    generate()
    build_features()
