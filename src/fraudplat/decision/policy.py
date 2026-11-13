"""Expected-cost decision policy (FR-08, BR-06). Business-owned configuration, separate from the model."""
from __future__ import annotations

import copy

import numpy as np

ACTIONS = ["approve", "step_up", "review", "decline"]


def _merge(base: dict, over: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in (over or {}).items():
        out[k] = _merge(out[k], v) if isinstance(v, dict) and isinstance(out.get(k), dict) else v
    return out


class DecisionPolicy:
    def __init__(self, cfg: dict, shadow_price: float | None = None, step_up_shadow: float | None = None):
        self.cfg = cfg
        self.version = cfg["version"]
        self.budget = cfg["review_rate_budget"]
        self.su_budget = cfg.get("step_up_rate_budget", 1.0)
        self.shadow = cfg.get("review_shadow_price", 0.0) if shadow_price is None else shadow_price
        self.shadow_su = cfg.get("step_up_shadow_price", 0.0) if step_up_shadow is None else step_up_shadow
        self.risk_limit = cfg["risk_limit_amount"]
        self.params = {"CNP": cfg["defaults"], "A2A": _merge(cfg["defaults"], cfg.get("segments", {}).get("A2A"))}

    def costs(self, p, amount, is_a2a, shadow: float | None = None, shadow_su: float | None = None) -> np.ndarray:
        """Expected cost of each action -> array [n, 4] in ACTIONS order."""
        p, amount, is_a2a = (np.atleast_1d(np.asarray(x, float)) for x in (p, amount, is_a2a))
        lam = self.shadow if shadow is None else shadow
        lam_su = self.shadow_su if shadow_su is None else shadow_su
        out = np.zeros((len(p), 4))
        for seg, mask in (("CNP", is_a2a == 0), ("A2A", is_a2a == 1)):
            if not mask.any():
                continue
            c, pp, a = self.params[seg], p[mask], amount[mask]
            loss = a * c["loss_multiplier"] + c["chargeback_fee"]
            su, rv = c["step_up"], c["review"]
            out[mask, 0] = pp * loss
            out[mask, 1] = (su["fixed"] + (1 - pp) * su["abandon_rate"] * (a * c["revenue_rate"] + c["false_fixed"])
                            + pp * (1 - su["effectiveness"]) * loss + lam_su)
            out[mask, 2] = (rv["fixed"] + (1 - pp) * rv["abandon_rate"] * (a * c["revenue_rate"] + c["false_fixed"])
                            + pp * (1 - rv["effectiveness"]) * loss + lam)
            out[mask, 3] = (1 - pp) * (c["false_fixed"] + a * c["revenue_rate"])
        return out

    def decide(self, p, amount, is_a2a, shadow: float | None = None, shadow_su: float | None = None) -> np.ndarray:
        return np.argmin(self.costs(p, amount, is_a2a, shadow, shadow_su), axis=1)

    def _bisect(self, rate_fn, budget, hi=500.0, iters=40) -> float:
        if rate_fn(0.0) <= budget:
            return 0.0
        lo = 0.0
        for _ in range(iters):
            mid = (lo + hi) / 2
            if rate_fn(mid) > budget:
                lo = mid
            else:
                hi = mid
        return float(hi)

    def calibrate_shadow_price(self, p, amount, is_a2a, rounds: int = 25, tol: float = 1e-3):
        """Calibrate review and step-up shadow prices so BOTH budgets hold. The two interact (a dearer review pushes
        cases into step-up and vice versa), so alternate until the prices stop moving. Returns (review, step_up)."""
        lam_r, lam_s = 0.0, 0.0
        for _ in range(rounds):
            new_r = self._bisect(lambda x: (self.decide(p, amount, is_a2a, x, lam_s) == 2).mean(), self.budget)
            new_s = self._bisect(lambda x: (self.decide(p, amount, is_a2a, new_r, x) == 1).mean(), self.su_budget)
            done = abs(new_r - lam_r) < tol and abs(new_s - lam_s) < tol
            lam_r, lam_s = new_r, new_s
            if done:
                break
        self.shadow, self.shadow_su = lam_r, lam_s
        return lam_r, lam_s

    def outcome_costs(self, actions, y, amount, is_a2a) -> dict:
        """Expected realised cost of a decision vector given ground truth (simulation only)."""
        a = np.asarray(amount, float)
        is_a2a = np.asarray(is_a2a, float)
        loss_fraud = np.zeros(len(a)); friction = np.zeros(len(a)); fp_cost = np.zeros(len(a))
        for seg, mask in (("CNP", is_a2a == 0), ("A2A", is_a2a == 1)):
            c = self.params[seg]
            loss = a[mask] * c["loss_multiplier"] + c["chargeback_fee"]
            act, yy = actions[mask], y[mask]
            eff = np.select([act == 0, act == 1, act == 2, act == 3], [0.0, c["step_up"]["effectiveness"],
                                                                         c["review"]["effectiveness"], 1.0])
            loss_fraud[mask] = yy * (1 - eff) * loss
            fix = np.select([act == 1, act == 2], [c["step_up"]["fixed"], c["review"]["fixed"]], 0.0)
            ab = np.select([act == 1, act == 2], [c["step_up"]["abandon_rate"], c["review"]["abandon_rate"]], 0.0)
            friction[mask] = fix
            fp_cost[mask] = (1 - yy) * (ab * (a[mask] * c["revenue_rate"] + ((act == 1) | (act == 2)) * c["false_fixed"])
                                        + (act == 3) * (c["false_fixed"] + a[mask] * c["revenue_rate"]))
        return {"fraud_loss": float(loss_fraud.sum()), "friction_cost": float(friction.sum()),
                "false_positive_cost": float(fp_cost.sum()), "total_cost": float(loss_fraud.sum() + friction.sum() + fp_cost.sum())}
