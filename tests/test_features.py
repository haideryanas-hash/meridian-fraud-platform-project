"""Leakage, label-delay and train/serve-skew tests (FR-03, FR-04)."""
import numpy as np
import pandas as pd

from fraudplat.config import DAY, LABEL_COLS
from fraudplat.features.build import replay
from fraudplat.features.engine import FEATURE_NAMES, WINDOWS, FeatureEngine

GRAPH_IDX = [i for i, n in enumerate(FEATURE_NAMES) if n.startswith("g_")]


class _NoLabels(dict):
    """Event dict that raises if the engine ever reads label / ground-truth fields while scoring."""

    def __getitem__(self, k):
        assert k not in LABEL_COLS, f"scoring path touched label field {k!r}"
        return super().__getitem__(k)


def test_scoring_path_never_reads_labels(small_events):
    from fraudplat.config import EVENT_COLS
    eng = FeatureEngine()
    for ev in small_events.iloc[:3000][EVENT_COLS].to_dict("records"):
        eng.process(_NoLabels(ev))


def test_replay_is_deterministic(small_events):
    ev = small_events.iloc[:5000]
    a, _ = replay(ev, verbose=False)
    b, _ = replay(ev, verbose=False)
    assert np.array_equal(a, b)


def test_future_truncation_does_not_change_past(small_events):
    """Features of row i may only depend on rows <= i: truncating the future must not change them."""
    ev = small_events
    T = 25 * DAY
    full, _ = replay(ev, verbose=False)
    cut = ev[ev["event_time"] < T].reset_index(drop=True)
    part, _ = replay(cut, verbose=False)
    assert np.array_equal(full[: len(cut)], part)


def test_label_delay_is_respected(small_events):
    """Changing labels that are not yet known at time T must not change any feature before T."""
    ev = small_events.copy()
    T = 25 * DAY
    base, _ = replay(ev, verbose=False)
    pre = (ev["event_time"] < T).to_numpy()

    # sensitivity check: the taint features must actually be exercised before T
    assert base[pre][:, GRAPH_IDX[:6]].sum() > 0

    flipped = ev.copy()
    late = flipped["label_time"] >= T
    flipped.loc[late, "is_fraud"] = 1 - flipped.loc[late, "is_fraud"]
    flipped.loc[late, ["taint_dev", "taint_sender", "label_observed"]] = 1
    alt, _ = replay(flipped, verbose=False)
    assert np.array_equal(base[pre], alt[pre])

    # negative control: if labels were visible immediately the test WOULD fail (proves test power)
    leaky = flipped.copy()
    leaky.loc[late, "label_time"] = leaky.loc[late, "event_time"]
    leak, _ = replay(leaky, verbose=False)
    assert not np.array_equal(base[pre], leak[pre])


def _ref_velocity(df, col, w):
    cnt, tot = np.zeros(len(df)), np.zeros(len(df))
    d = df.reset_index(drop=True)
    for _, g in d[d[col] >= 0].groupby(col, sort=False):
        times, am, idx = g["event_time"].to_numpy(), g["amount"].to_numpy(), g.index.to_numpy()
        cum = np.concatenate([[0.0], np.cumsum(am)])
        lo = np.searchsorted(times, times - w, side="left")
        pos = np.arange(len(g))
        cnt[idx], tot[idx] = pos - lo, cum[pos] - cum[lo]
    return cnt, tot


def test_online_matches_independent_offline_reference(small_events):
    """Training/serving skew test: engine output == independent vectorised pandas implementation."""
    ev = small_events.iloc[:8000].reset_index(drop=True)
    X, _ = replay(ev, verbose=False)
    ix = {n: i for i, n in enumerate(FEATURE_NAMES)}
    for short, col in [("cust", "customer_id"), ("dev", "device_id"), ("ip", "ip_id"), ("ben", "beneficiary_id")]:
        for wname, w in WINDOWS:
            c, s = _ref_velocity(ev, col, w)
            assert np.allclose(X[:, ix[f"{short}_cnt_{wname}"]], c), (short, wname, "cnt")
            assert np.allclose(X[:, ix[f"{short}_sum_{wname}"]], s, rtol=1e-4, atol=1e-2), (short, wname, "sum")
    prior = ev.groupby("customer_id").cumcount().to_numpy()
    assert np.array_equal(X[:, ix["cust_prior_txns"]], prior)
    last = ev.groupby("customer_id")["event_time"].shift(1)
    since = np.minimum((ev["event_time"] - last).fillna(30 * DAY).to_numpy(), 30 * DAY)
    assert np.allclose(X[:, ix["secs_since_last"]], since)


def test_engine_requires_ordered_events(small_events):
    import pytest
    bad = small_events.iloc[:50].iloc[::-1]
    with pytest.raises(AssertionError):
        replay(bad, verbose=False)
