"""Drift (PSI / KS) and fairness screening reports (FR-14, BR-10)."""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..evaluation.metrics import ks, psi


def drift_report(X_ref: np.ndarray, X_cur_by_period: dict[str, np.ndarray], names: list[str],
                 score_ref: np.ndarray, score_cur_by_period: dict[str, np.ndarray], alert: float = 0.2) -> dict:
    rows = []
    for per, Xc in X_cur_by_period.items():
        for j, n in enumerate(names):
            ref, cur = X_ref[:, j], Xc[:, j]
            if np.ptp(ref) == 0:
                continue
            rows.append({"period": per, "feature": n, "psi": psi(ref, cur), "ks": ks(ref[::5], cur[::5])})
    feat = pd.DataFrame(rows)
    score = [{"period": per, "psi": psi(score_ref, s), "ks": ks(score_ref, s)} for per, s in score_cur_by_period.items()]
    worst = feat.sort_values("psi", ascending=False).groupby("period").head(5)
    return {"psi_alert_threshold": alert,
            "features_over_threshold": feat[feat.psi > alert].groupby("period").size().to_dict(),
            "n_features_checked": int(feat.feature.nunique()),
            "worst_features": worst.round(4).to_dict("records"),
            "score_drift": score,
            "score_alerts": [r["period"] for r in score if r["psi"] > alert]}


def fairness_report(df: pd.DataFrame, segment_cols: list[str], ratio_alert: float = 1.25, min_n: int = 500) -> dict:
    """df columns: action (0..3), is_fraud, plus segment columns. Rates of friction applied to LEGITIMATE payments,
    plus fraud capture, per segment; flags segments whose friction rate exceeds ratio_alert x the overall rate."""
    legit = df[df.is_fraud == 0]
    overall = {"decline_rate": float((df.action == 3).mean()), "friction_rate_legit": float((legit.action > 0).mean()),
               "decline_rate_legit": float((legit.action == 3).mean())}
    out = {"overall": overall, "segments": [], "flags": []}
    for col in segment_cols:
        for val, g in df.groupby(col):
            if len(g) < min_n:
                continue
            gl = g[g.is_fraud == 0]
            row = {"segment": col, "value": int(val), "n": int(len(g)), "decline_rate": float((g.action == 3).mean()),
                   "friction_rate_legit": float((gl.action > 0).mean()), "decline_rate_legit": float((gl.action == 3).mean()),
                   "fraud_recall": float((g[g.is_fraud == 1].action > 0).mean()) if (g.is_fraud == 1).any() else None}
            row["friction_ratio"] = row["friction_rate_legit"] / max(overall["friction_rate_legit"], 1e-9)
            out["segments"].append(row)
            if row["friction_ratio"] > ratio_alert:
                out["flags"].append({k: row[k] for k in ("segment", "value", "n", "friction_ratio")})
    return out
