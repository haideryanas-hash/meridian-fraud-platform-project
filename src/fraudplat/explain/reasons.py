"""Reason codes (FR-09, BR-05): TreeSHAP contributions mapped to a controlled taxonomy."""
from __future__ import annotations

from fnmatch import fnmatch

import numpy as np


class ReasonCoder:
    def __init__(self, taxonomy: dict, feature_names: list[str], extra_names=("anomaly_score", "gnn_send", "gnn_ben", "gnn_dev")):
        self.tax = taxonomy
        self.names = list(feature_names) + list(extra_names)
        self.code_of = {}
        for n in self.names:
            self.code_of[n] = next((c for c, d in taxonomy.items() if any(fnmatch(n, pat) for pat in d["features"])), "RC99")
        self.code_of["gnn_send"] = self.code_of["gnn_ben"] = self.code_of["gnn_dev"] = "RC11"
        self.code_of["anomaly_score"] = "RC10"

    def explain(self, contribs: np.ndarray, top: int = 3, min_share: float = 0.0):
        """contribs: logit-scale contributions aligned with self.names. Returns positive (risk-raising) codes."""
        agg: dict[str, float] = {}
        top_feat: dict[str, tuple[str, float]] = {}
        for n, v in zip(self.names, contribs):
            if v <= 0:
                continue
            c = self.code_of[n]
            agg[c] = agg.get(c, 0.0) + float(v)
            if c not in top_feat or v > top_feat[c][1]:
                top_feat[c] = (n, float(v))
        total = sum(agg.values())
        ranked = sorted(agg.items(), key=lambda kv: -kv[1])[:top]
        out = [{"code": c, "text": self.tax[c]["text"], "contribution": round(v, 4), "top_feature": top_feat[c][0]}
               for c, v in ranked if total and v / total >= min_share]
        return out or [{"code": "RC99", "text": self.tax["RC99"]["text"], "contribution": 0.0, "top_feature": ""}]
