"""Safe experimentation (BR-08, FR-13): shadow, champion/challenger and canary routing with automatic rollback."""
from __future__ import annotations

import hashlib
from collections import deque

import numpy as np


class TrafficRouter:
    """Deterministic, hash-based routing by transaction id (replayable; the same txn always takes the same arm).

    mode "off"    : champion decides everything
    mode "shadow" : champion decides; challenger also scores (decisions ignored, only compared)
    mode "canary" : `pct` of traffic (optionally only listed segments) is decided by the challenger
    """

    def __init__(self, mode: str = "off", pct: float = 0.0, segments: set[str] | None = None, salt: str = "rel-1"):
        assert mode in ("off", "shadow", "canary")
        self.mode, self.pct, self.segments, self.salt = mode, pct, segments, salt

    def _u(self, txn_id) -> float:
        return int(hashlib.sha256(f"{self.salt}:{txn_id}".encode()).hexdigest()[:8], 16) / 2 ** 32

    def route(self, txn_id, segment: str = "all") -> dict:
        in_seg = self.segments is None or segment in self.segments
        if self.mode == "canary" and in_seg and self._u(txn_id) < self.pct:
            return {"decide": "challenger", "score_with": ["champion", "challenger"]}
        if self.mode == "shadow" and in_seg and self._u(txn_id) < max(self.pct, 1e-12 if self.pct == 0 else self.pct):
            return {"decide": "champion", "score_with": ["champion", "challenger"]}
        return {"decide": "champion", "score_with": ["champion"]}


DEFAULT_GUARDRAILS = {"decline_rate_max_ratio": 1.25, "review_rate_max": 0.01, "p99_latency_ms": 100.0,
                      "expected_loss_max_ratio": 1.10, "scoring_fallback_rate_max": 0.05}


class GuardrailMonitor:
    """Rolling-window comparison of the canary arm against the champion arm; breach => automatic rollback."""

    def __init__(self, guardrails: dict | None = None, window: int = 3000, min_n: int = 300):
        self.g = {**DEFAULT_GUARDRAILS, **(guardrails or {})}
        self.min_n = min_n
        self.arms = {a: deque(maxlen=window) for a in ("champion", "challenger")}

    def record(self, arm: str, decision: str, latency_ms: float, amount: float, p: float | None, mode: str = "full"):
        self.arms[arm].append((decision, latency_ms, amount, p if p is not None else 0.0, mode))

    @staticmethod
    def _stats(rows) -> dict:
        n = len(rows)
        dec = [r[0] for r in rows]
        amt = np.array([r[2] for r in rows]) if n else np.array([1.0])
        pr = np.array([r[3] for r in rows]) if n else np.array([0.0])
        appr = np.array([d == "approve" for d in dec]) if n else np.array([True])
        return {"n": n, "decline_rate": dec.count("decline") / max(n, 1), "review_rate": dec.count("review") / max(n, 1),
                "p99_ms": float(np.percentile([r[1] for r in rows], 99)) if n else 0.0,
                "expected_loss_rate": float((amt * pr * appr).sum() / max(amt.sum(), 1e-9)),
                "fallback_rate": sum(1 for r in rows if r[4] != "full") / max(n, 1)}

    def check(self) -> dict:
        ch, cp = self._stats(self.arms["challenger"]), self._stats(self.arms["champion"])
        if ch["n"] < self.min_n or cp["n"] < self.min_n:
            return {"breach": [], "rollback": False, "reason": "insufficient data", "challenger": ch, "champion": cp}
        b = []
        if ch["decline_rate"] > self.g["decline_rate_max_ratio"] * max(cp["decline_rate"], 1e-4):
            b.append("decline_rate")
        if ch["review_rate"] > self.g["review_rate_max"]:
            b.append("review_rate")
        if ch["p99_ms"] > self.g["p99_latency_ms"]:
            b.append("p99_latency")
        if ch["expected_loss_rate"] > self.g["expected_loss_max_ratio"] * max(cp["expected_loss_rate"], 1e-9):
            b.append("expected_loss")
        if ch["fallback_rate"] > self.g["scoring_fallback_rate_max"]:
            b.append("fallback_rate")
        return {"breach": b, "rollback": bool(b), "challenger": ch, "champion": cp}


class RolloutController:
    def __init__(self, router: TrafficRouter, monitor: GuardrailMonitor):
        self.router, self.monitor, self.history = router, monitor, []

    def observe(self, arm, decision, latency_ms, amount, p, mode="full"):
        self.monitor.record(arm, decision, latency_ms, amount, p, mode)
        res = self.monitor.check()
        if res["rollback"] and self.router.mode != "off":
            self.history.append({"action": "rollback", "from_mode": self.router.mode, "breach": res["breach"]})
            self.router.mode = "off"
        return res
