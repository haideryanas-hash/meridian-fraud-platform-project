"""Tier 4: stacked ensemble + isotonic calibration with an uncertainty band (FR-07)."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression

from ..graph.gnn import UNKNOWN_LOGIT

STACK_COLS = ["gbdt_margin", "anomaly", "gnn_send", "gnn_ben", "gnn_dev"]


def stack_inputs(margin, anomaly, gnn_send, gnn_ben, gnn_dev, cols=STACK_COLS) -> np.ndarray:
    """Columns stacker is fed. GNN logits are shifted so 'unknown node' = 0 (no spurious offset for CNP rows)."""
    a = lambda v: np.atleast_1d(np.asarray(v, float))  # noqa: E731  (accepts scalars, lists and arrays)
    allc = {"gbdt_margin": a(margin), "anomaly": a(anomaly), "gnn_send": a(gnn_send) - UNKNOWN_LOGIT,
            "gnn_ben": a(gnn_ben) - UNKNOWN_LOGIT, "gnn_dev": a(gnn_dev) - UNKNOWN_LOGIT}
    return np.column_stack([allc[c] for c in cols])


class Stacker:
    def __init__(self, cols=STACK_COLS, C=1.0):
        self.cols, self.lr = list(cols), LogisticRegression(C=C, max_iter=1000)
        self.mean_ = None

    def fit(self, Z, y):
        self.lr.fit(Z, y)
        self.mean_ = Z.mean(0)
        return self

    def logit(self, Z):
        return self.lr.decision_function(Z)

    @property
    def coef(self):
        return dict(zip(self.cols, self.lr.coef_[0]))


class Calibrator:
    """Isotonic calibration on the most recent mature window + Wilson interval per isotonic plateau."""

    FLOOR = 2e-5

    def fit(self, s, y):
        self.iso = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0).fit(s, y)
        p = np.round(self.iso.predict(s), 12)
        g = pd.DataFrame({"p": p, "y": y}).groupby("p")["y"].agg(["sum", "count"]).reset_index()
        n, ph, z = g["count"].to_numpy(float), g["sum"].to_numpy(float) / g["count"].to_numpy(float), 1.96
        den = 1 + z * z / n
        centre = (ph + z * z / (2 * n)) / den
        half = z * np.sqrt(ph * (1 - ph) / n + z * z / (4 * n * n)) / den
        self.p_vals = g["p"].to_numpy()
        self.lo = np.clip(centre - half, 0, 1)
        self.hi = np.clip(centre + half, 0, 1)
        return self

    def predict(self, s):
        return np.clip(self.iso.predict(s), self.FLOOR, 1.0)

    def predict_with_band(self, s):
        p = np.round(self.iso.predict(s), 12)
        i = np.clip(np.searchsorted(self.p_vals, p), 0, len(self.p_vals) - 1)
        return np.clip(p, self.FLOOR, 1.0), np.clip(self.lo[i], 0, 1), np.clip(self.hi[i], 0, 1)
