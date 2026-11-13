"""Evaluation metrics (BRD section 3 / 12): PR-AUC, recall at fixed review rate, dollar-weighted recall,
precision at matched recall, calibration (ECE, Brier), PSI / KS drift, paired bootstrap confidence intervals."""
from __future__ import annotations

import numpy as np
from scipy.stats import ks_2samp
from sklearn.metrics import average_precision_score, roc_auc_score


def top_rate_mask(score, rate):
    k = max(int(round(len(score) * rate)), 1)
    idx = np.argpartition(-score, k - 1)[:k]
    m = np.zeros(len(score), bool)
    m[idx] = True
    return m


def recall_at_rate(y, score, rate):
    return float(y[top_rate_mask(score, rate)].sum() / max(y.sum(), 1))


def precision_at_rate(y, score, rate):
    return float(y[top_rate_mask(score, rate)].mean())


def dollar_recall_at_rate(y, score, amount, rate):
    m = top_rate_mask(score, rate)
    return float((amount * y)[m].sum() / max((amount * y).sum(), 1e-9))


def precision_at_recall(y, score, target_recall):
    order = np.argsort(-score, kind="stable")
    tp = np.cumsum(y[order])
    k = int(np.searchsorted(tp, target_recall * y.sum(), side="left")) + 1
    return float(tp[min(k, len(tp)) - 1] / min(k, len(tp)))


def ece(y, p, bins=10):
    """Expected calibration error with equal-mass bins."""
    order = np.argsort(p)
    err, n = 0.0, len(p)
    for chunk in np.array_split(order, bins):
        err += len(chunk) / n * abs(p[chunk].mean() - y[chunk].mean())
    return float(err)


def brier(y, p):
    return float(np.mean((p - y) ** 2))


def reliability_table(y, p, bins=10):
    order = np.argsort(p)
    rows = []
    for chunk in np.array_split(order, bins):
        rows.append({"mean_pred": float(p[chunk].mean()), "obs_rate": float(y[chunk].mean()), "n": int(len(chunk))})
    return rows


def summary(y, score, amount, rate=0.01):
    return {"pr_auc": float(average_precision_score(y, score)), "roc_auc": float(roc_auc_score(y, score)),
            "recall_at_1pct": recall_at_rate(y, score, rate), "precision_at_1pct": precision_at_rate(y, score, rate),
            "dollar_recall_at_1pct": dollar_recall_at_rate(y, score, amount, rate)}


def paired_bootstrap(y, a, b, amount, rate=0.01, n_boot=200, seed=0):
    """95% CI of metric(a)-metric(b) over resampled test sets (a = new model, b = baseline)."""
    rng = np.random.default_rng(seed)
    n = len(y)
    out = {"pr_auc": [], "recall_at_1pct": []}
    for _ in range(n_boot):
        i = rng.integers(0, n, n)
        if y[i].sum() == 0:
            continue
        out["pr_auc"].append(average_precision_score(y[i], a[i]) - average_precision_score(y[i], b[i]))
        out["recall_at_1pct"].append(recall_at_rate(y[i], a[i], rate) - recall_at_rate(y[i], b[i], rate))
    return {k: [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))] for k, v in out.items()}


def bootstrap_ci(y, score, fn, n_boot=200, seed=0):
    rng = np.random.default_rng(seed)
    vals = []
    n = len(y)
    for _ in range(n_boot):
        i = rng.integers(0, n, n)
        if y[i].sum():
            vals.append(fn(y[i], score[i]))
    return [float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))]


def psi(ref, cur, bins=10, eps=1e-4):
    """Population Stability Index with quantile bins from the reference sample."""
    ref, cur = np.asarray(ref, float), np.asarray(cur, float)
    edges = np.unique(np.quantile(ref, np.linspace(0, 1, bins + 1)))
    if len(edges) < 3:
        return 0.0
    edges[0], edges[-1] = -np.inf, np.inf
    r = np.histogram(ref, edges)[0] / len(ref) + eps
    c = np.histogram(cur, edges)[0] / len(cur) + eps
    return float(np.sum((c - r) * np.log(c / r)))


def ks(ref, cur):
    return float(ks_2samp(ref, cur).statistic)
