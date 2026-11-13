"""Real-time scorer: features -> GBDT + GNN + anomaly -> stacker -> calibrated p (+band) -> policy -> reason codes.

Resilience (BR-07 / NFR-04): full ensemble -> lightweight tabular GBDT -> rules. The scorer never returns an
unscored approval above the configured risk limit. Every decision is written to a tamper-evident, hash-chained
audit log with the exact feature vector so it can be replayed (FR-12, NFR-07).
"""
from __future__ import annotations

import fnmatch
import hashlib
import json
import logging
import threading
import time
from collections import deque
from pathlib import Path

import numpy as np

from ..config import load_yaml
from ..decision.policy import ACTIONS, DecisionPolicy
from ..explain.reasons import ReasonCoder
from ..features.engine import FEATURE_NAMES, TABULAR_FEATURES, FeatureEngine
from ..graph.gnn import UNKNOWN_LOGIT
from ..models.bundle import ModelBundle
from ..models.ensemble import stack_inputs
from ..models.gbdt import margin, sigmoid

log = logging.getLogger("fraudplat.scorer")

_TAB_IDX = [FEATURE_NAMES.index(n) for n in TABULAR_FEATURES]

# Features that do not exist for a channel are 0/"missing"; trees can still attribute risk to that absence
# (e.g. "no merchant" correlates with A2A fraud). Such attributions must never surface as customer/audit reasons.
_NOT_APPLICABLE = {1: ("mer_*", "card_*", "merchant_cat", "new_mer_for_cust", "bin_risk"),                 # A2A: no merchant/card
                   0: ("ben_*", "new_ben_for_cust", "g_ben_*", "gnn_ben")}                                 # CNP: no beneficiary


class CircuitBreaker:
    """Opens after `threshold` consecutive failures/slow calls; half-opens after `reset_s`."""

    def __init__(self, threshold: int = 5, reset_s: float = 30.0):
        self.threshold, self.reset_s, self.fails, self.opened_at = threshold, reset_s, 0, None

    def allow(self) -> bool:
        if self.opened_at is None:
            return True
        if time.monotonic() - self.opened_at >= self.reset_s:
            self.opened_at, self.fails = None, self.threshold - 1       # half-open: one trial call
            return True
        return False

    def record(self, ok: bool):
        if ok:
            self.fails, self.opened_at = 0, None
        else:
            self.fails += 1
            if self.fails >= self.threshold:
                self.opened_at = time.monotonic()


class AuditLog:
    """Append-only JSONL, each record carries sha256(prev_hash + payload): any edit/removal breaks the chain."""

    def __init__(self, path: str | Path | None):
        self.path, self.prev, self.lock = (Path(path) if path else None), "0" * 64, threading.Lock()
        if self.path and self.path.exists():
            self.path.unlink()

    def append(self, record: dict):
        if not self.path:
            return
        with self.lock:
            payload = json.dumps(record, sort_keys=True)
            h = hashlib.sha256((self.prev + payload).encode()).hexdigest()
            with open(self.path, "a") as f:
                f.write(json.dumps({"prev": self.prev, "hash": h, "record": record}, sort_keys=True) + "\n")
            self.prev = h

    @staticmethod
    def verify(path: str | Path) -> tuple[bool, int]:
        prev, n = "0" * 64, 0
        for line in open(path):
            row = json.loads(line)
            h = hashlib.sha256((prev + json.dumps(row["record"], sort_keys=True)).encode()).hexdigest()
            if row["prev"] != prev or row["hash"] != h:
                return False, n
            prev, n = h, n + 1
        return True, n


class Scorer:
    def __init__(self, bundle: ModelBundle, engine: FeatureEngine, policy_cfg: dict | None = None, taxonomy: dict | None = None,
                 audit_path: str | Path | None = None, slow_ms: float = 80.0):
        self.b, self.engine = bundle, engine
        self.policy = DecisionPolicy(policy_cfg or load_yaml("config/policy.yaml"), bundle.shadow_price, bundle.step_up_shadow)
        self.coder = ReasonCoder(taxonomy or load_yaml("config/reason_codes.yaml"), FEATURE_NAMES)
        self.audit = AuditLog(audit_path)
        self.breaker, self.slow_ms = CircuitBreaker(), slow_ms
        self.faults: set[str] = set()                      # test hook: {"full", "light"} to simulate tier outages
        self.latencies: list[float] = []
        self.modes: dict[str, int] = {}
        self.errors: dict[str, int] = {}                   # tier -> count of real (non-injected) failures
        self.last_errors: deque = deque(maxlen=5)
        self._applicable = {ch: np.array([not any(fnmatch.fnmatch(n, pat) for pat in pats) for n in self.coder.names], float)
                            for ch, pats in _NOT_APPLICABLE.items()}
        self._coef = bundle.stacker.lr.coef_[0]            # order: STACK_COLS
        self._zmean = bundle.stacker.mean_

    # ------------------------------------------------------------------ tiers
    def _full(self, x: np.ndarray, ev: dict):
        if "full" in self.faults:
            raise RuntimeError("full tier unavailable (injected)")
        m = float(margin(self.b.gbdt_full, x)[0])
        an = float(self.b.anomaly.score(x)[0])
        gs, gb, gd = (float(v[0]) for v in self.b.gnn_store.lookup(ev["event_time"], ev["customer_id"],
                                                                   ev["beneficiary_id"], ev["device_id"]))
        return self._finish_full(m, an, gs, gb, gd), {"anomaly": an, "gnn": [gs, gb, gd]}

    def _finish_full(self, m, an, gs, gb, gd):
        Z = stack_inputs([m], [an], [gs], [gb], [gd])
        pr = sigmoid(self.b.stacker.logit(Z))
        p, lo, hi = (float(v[0]) for v in self.b.calibrator.predict_with_band(pr))
        return p, lo, hi, Z[0]

    def _light(self, x: np.ndarray):
        if "light" in self.faults:
            raise RuntimeError("light tier unavailable (injected)")
        pr = sigmoid(margin(self.b.gbdt_light, x[:, _TAB_IDX]))
        p, lo, hi = (float(v[0]) for v in self.b.calibrator_light.predict_with_band(pr))
        return p, lo, hi

    def _rules(self, ev: dict):
        """Last-resort rules. Never approves an unscored payment above the risk limit."""
        if ev["amount"] >= self.policy.risk_limit or (ev["ip_rep"] > .6 and ev["amount"] > 250):
            return 2
        if ev["ip_country"] != ev["home_country"] and ev["amount"] > 400:
            return 1
        return 0

    def _note_error(self, tier: str, exc: Exception):
        injected = "injected" in str(exc) or str(exc) == "circuit open"
        if not injected:
            self.errors[tier] = self.errors.get(tier, 0) + 1
            self.last_errors.append(f"{tier}: {type(exc).__name__}: {exc}")
            if self.errors[tier] == 1:
                log.exception("scoring tier %s failed; falling back", tier)

    # ------------------------------------------------------------------ reason codes
    def _reasons(self, mode: str, x: np.ndarray, Z, is_a2a: int = 0):
        names = self.coder.names
        contrib = np.zeros(len(names))
        if mode == "full":
            c = self.b.gbdt_full.predict(x, pred_contrib=True, num_iteration=self.b.gbdt_full.best_iteration or None)[0]
            contrib[: len(FEATURE_NAMES)] = c[:-1] * self._coef[0]
            extra = self._coef[1:] * (Z[1:] - self._zmean[1:])         # anomaly, gnn_send, gnn_ben, gnn_dev
            contrib[len(FEATURE_NAMES):] = extra
        elif mode == "light":
            c = self.b.gbdt_light.predict(x[:, _TAB_IDX], pred_contrib=True, num_iteration=self.b.gbdt_light.best_iteration or None)[0]
            for j, i in enumerate(_TAB_IDX):
                contrib[i] = c[j]
        else:
            return [{"code": "RC99", "text": self.coder.tax["RC99"]["text"], "contribution": 0.0, "top_feature": "rules_fallback"}]
        return self.coder.explain(contrib * self._applicable[int(is_a2a)])

    # ------------------------------------------------------------------ main entry
    def score(self, ev: dict, log_audit: bool = True) -> dict:
        t0 = time.perf_counter()
        self.engine.release(ev["event_time"])
        x = np.asarray(self.engine.compute(ev), np.float32)[None, :]
        mode, p, lo, hi, Z, aux, action = "full", None, None, None, None, {}, None
        try:
            if not self.breaker.allow():
                raise RuntimeError("circuit open")
            (p, lo, hi, Z), aux = self._full(x, ev)
            self.breaker.record((time.perf_counter() - t0) * 1000 < self.slow_ms)
        except Exception as exc:                            # never let a tier failure block a payment, but never hide it either
            self._note_error("full", exc)
            self.breaker.record(False)
            mode = "light"
            try:
                p, lo, hi = self._light(x)
            except Exception as exc2:
                self._note_error("light", exc2)
                mode = "rules"
        if mode == "rules":
            action = self._rules(ev)
        else:
            action = int(self.policy.decide([p], [ev["amount"]], [ev["is_a2a"]])[0])
        reasons = self._reasons(mode, x, Z, ev["is_a2a"]) if action != 0 else []
        self.engine.update(ev)                              # state always advances, whichever tier answered
        ms = (time.perf_counter() - t0) * 1000
        self.latencies.append(ms)
        self.modes[mode] = self.modes.get(mode, 0) + 1
        out = {"txn_id": ev.get("txn_id"), "decision": ACTIONS[action], "fraud_probability": p,
               "probability_band": [lo, hi] if lo is not None else None, "reason_codes": reasons, "scoring_mode": mode,
               "model_version": self.b.version, "policy_version": self.policy.version, "latency_ms": round(ms, 3)}
        if log_audit:
            self.audit.append({"txn_id": ev.get("txn_id"), "event_time": ev["event_time"], "amount": ev["amount"],
                               "is_a2a": ev["is_a2a"], "ip_rep": ev["ip_rep"], "ip_country": ev["ip_country"],
                               "home_country": ev["home_country"], "features": x[0].tolist(), "aux": aux, "mode": mode,
                               "model_version": self.b.version, "policy_version": self.policy.version,
                               "shadow": [self.policy.shadow, self.policy.shadow_su], "p": p, "band": [lo, hi],
                               "action": ACTIONS[action], "reason_codes": [r["code"] for r in reasons]})
        return out

    def latency_summary(self) -> dict:
        a = np.array(self.latencies) if self.latencies else np.array([0.0])
        return {"n": len(self.latencies), "p50_ms": float(np.percentile(a, 50)), "p95_ms": float(np.percentile(a, 95)),
                "p99_ms": float(np.percentile(a, 99)), "max_ms": float(a.max()), "modes": dict(self.modes),
                "tier_errors": dict(self.errors), "last_errors": list(self.last_errors)}


def replay_record(record: dict, bundle: ModelBundle, policy_cfg: dict | None = None) -> dict:
    """Re-derive a logged decision from its logged inputs and model/policy version (audit replay)."""
    pol = DecisionPolicy(policy_cfg or load_yaml("config/policy.yaml"), record["shadow"][0], record["shadow"][1])
    x = np.asarray(record["features"], np.float32)[None, :]
    if record["mode"] == "full":
        gs, gb, gd = record["aux"]["gnn"]
        m = float(margin(bundle.gbdt_full, x)[0])
        Z = stack_inputs([m], [record["aux"]["anomaly"]], [gs], [gb], [gd])
        p, _, _ = bundle.calibrator.predict_with_band(sigmoid(bundle.stacker.logit(Z)))
        p = float(p[0])
    elif record["mode"] == "light":
        p = float(bundle.calibrator_light.predict_with_band(sigmoid(margin(bundle.gbdt_light, x[:, _TAB_IDX])))[0][0])
    else:
        return {"action": record["action"], "p": None}
    return {"action": ACTIONS[int(pol.decide([p], [record["amount"]], [record["is_a2a"]])[0])], "p": p}