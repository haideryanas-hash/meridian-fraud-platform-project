"""Versioned model bundle (a minimal file-based registry; swap for MLflow in production)."""
from __future__ import annotations

import json
from pathlib import Path

import joblib
import lightgbm as lgb
import numpy as np

from ..graph.gnn import GNNStore


class ModelBundle:
    def __init__(self, feature_names, gbdt_full, gbdt_light, anomaly, stacker, calibrator, gnn_store,
                 shadow_price, meta, step_up_shadow=0.0, calibrator_light=None):
        self.feature_names, self.gbdt_full, self.gbdt_light = list(feature_names), gbdt_full, gbdt_light
        self.anomaly, self.stacker, self.calibrator, self.gnn_store = anomaly, stacker, calibrator, gnn_store
        self.shadow_price, self.step_up_shadow, self.meta = shadow_price, step_up_shadow, meta
        self.calibrator_light = calibrator_light
        self.version = meta["model_version"]

    def save(self, d: str | Path):
        d = Path(d)
        d.mkdir(parents=True, exist_ok=True)
        self.gbdt_full.save_model(str(d / "gbdt_full.txt"), num_iteration=self.gbdt_full.best_iteration or -1)
        self.gbdt_light.save_model(str(d / "gbdt_light.txt"), num_iteration=self.gbdt_light.best_iteration or -1)
        joblib.dump({"anomaly": self.anomaly, "stacker": self.stacker, "calibrator": self.calibrator,
                     "calibrator_light": self.calibrator_light}, d / "tiers.joblib")
        self.gnn_store.save(d / "gnn_store.npz")
        (d / "runtime.json").write_text(json.dumps({"feature_names": self.feature_names,
                                                    "review_shadow_price": self.shadow_price, "step_up_shadow_price": self.step_up_shadow, "meta": self.meta}, indent=2))

    @classmethod
    def load(cls, d: str | Path):
        d = Path(d)
        rt = json.loads((d / "runtime.json").read_text())
        t = joblib.load(d / "tiers.joblib")
        return cls(rt["feature_names"], lgb.Booster(model_file=str(d / "gbdt_full.txt")),
                   lgb.Booster(model_file=str(d / "gbdt_light.txt")), t["anomaly"], t["stacker"], t["calibrator"],
                   GNNStore.load(d / "gnn_store.npz"), rt["review_shadow_price"], rt["meta"],
                   rt.get("step_up_shadow_price", 0.0), t.get("calibrator_light"))
