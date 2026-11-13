"""Tier 1: gradient-boosted trees (LightGBM), time-based early stopping on PR-AUC, monotone constraints."""
from __future__ import annotations

import lightgbm as lgb
import numpy as np

# risk-intuitive features: more known-bad links can never LOWER the risk score
MONOTONE_UP = {"g_dev_taint", "g_ip_taint", "g_ben_taint", "g_acct_taint", "g_ben_dev_taint", "g_nbr_taint"}

PARAMS = dict(objective="binary", learning_rate=0.05, num_leaves=48, min_child_samples=40, feature_fraction=0.7,
              bagging_fraction=0.8, bagging_freq=1, lambda_l2=5.0, scale_pos_weight=4.0, max_bin=127,
              n_jobs=1, deterministic=True, force_row_wise=True, seed=42, verbose=-1, metric="average_precision")


def train_gbdt(X_tr, y_tr, X_es, y_es, names, n_rounds=800, params=None, monotone=True):
    p = dict(PARAMS, **(params or {}))
    if monotone:
        p["monotone_constraints"] = [1 if n in MONOTONE_UP else 0 for n in names]
    dtr = lgb.Dataset(X_tr, y_tr, feature_name=list(names), free_raw_data=False)
    des = lgb.Dataset(X_es, y_es, reference=dtr)
    return lgb.train(p, dtr, n_rounds, valid_sets=[des], callbacks=[lgb.early_stopping(40, verbose=False)])


def margin(booster: lgb.Booster, X) -> np.ndarray:
    return booster.predict(X, raw_score=True, num_iteration=booster.best_iteration or None)


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))
