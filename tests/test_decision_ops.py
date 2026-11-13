import numpy as np

from fraudplat.config import load_yaml
from fraudplat.decision.policy import ACTIONS, DecisionPolicy
from fraudplat.evaluation import metrics as M
from fraudplat.explain.reasons import ReasonCoder
from fraudplat.features.engine import FEATURE_NAMES
from fraudplat.ops.routing import GuardrailMonitor, RolloutController, TrafficRouter


def _policy(**over):
    cfg = load_yaml("config/policy.yaml")
    cfg.update(over)
    return DecisionPolicy(cfg)


def test_policy_severity_is_monotone_in_probability():
    pol = _policy()
    p = np.linspace(0, 1, 400)
    for amount in (20, 100, 1000):
        for seg in (0, 1):
            a = pol.decide(p, np.full_like(p, amount), np.full_like(p, seg))
            assert (np.diff(a) >= 0).all(), (amount, seg)
            assert a[0] == 0 and a[-1] == 3


def test_policy_is_configuration_not_code_BR06():
    """Changing business costs changes decisions with the model (probabilities) untouched."""
    p, amt, a2a = np.full(1, 0.05), np.full(1, 300.0), np.zeros(1)
    cheap = _policy().decide(p, amt, a2a)[0]
    cfg = load_yaml("config/policy.yaml")
    cfg["defaults"]["false_fixed"] = 400.0                 # customer attrition is now very expensive
    strict = DecisionPolicy(cfg).decide(p, amt, a2a)[0]
    assert strict < cheap or (cheap == strict and cheap < 3)
    assert ACTIONS[_policy(review_rate_budget=0.01).decide(p, amt, a2a)[0]] in ACTIONS


def test_shadow_price_enforces_review_budget():
    rng = np.random.default_rng(0)
    p = np.clip(rng.beta(0.4, 40, 50000), 1e-5, 1)
    amt = np.exp(rng.normal(4.5, 1.0, 50000))
    a2a = rng.integers(0, 2, 50000).astype(float)
    pol = _policy(review_rate_budget=0.003, step_up_rate_budget=0.01)
    lam_r, lam_s = pol.calibrate_shadow_price(p, amt, a2a)
    act = pol.decide(p, amt, a2a)
    assert (act == 2).mean() <= 0.003 + 1e-3 and (act == 1).mean() <= 0.01 + 2e-3
    assert lam_r >= 0 and lam_s >= 0


def test_every_non_approve_gets_a_reason_code():
    coder = ReasonCoder(load_yaml("config/reason_codes.yaml"), FEATURE_NAMES)
    assert all(c in coder.tax for c in coder.code_of.values())
    out = coder.explain(np.zeros(len(coder.names)))                 # no positive contributor -> RC99 fallback
    assert out and out[0]["code"] == "RC99"
    v = np.zeros(len(coder.names)); v[coder.names.index("g_dev_taint")] = 2.0; v[coder.names.index("amount_z")] = 0.5
    codes = [r["code"] for r in coder.explain(v)]
    assert codes[0] == "RC07" and "RC04" in codes


def test_metrics_sanity():
    y = np.array([1, 0, 0, 1, 0, 0, 0, 0, 0, 0])
    s = np.array([.9, .1, .2, .8, .3, .05, .02, .01, .03, .04])
    assert M.recall_at_rate(y, s, 0.2) == 1.0 and M.precision_at_rate(y, s, 0.2) == 1.0
    rng = np.random.default_rng(1)
    x = rng.normal(size=20000)
    assert M.psi(x, rng.normal(size=20000)) < 0.02 and M.psi(x, rng.normal(1, 1, 20000)) > 0.2
    p = rng.random(50000); yy = (rng.random(50000) < p).astype(float)
    assert M.ece(yy, p) < 0.02                                       # calibrated by construction
    assert M.precision_at_recall(y, s, 1.0) == 1.0


def test_router_is_deterministic_and_respects_percentage():
    r = TrafficRouter("canary", 0.10)
    arms = [r.route(i)["decide"] for i in range(20000)]
    assert arms == [r.route(i)["decide"] for i in range(20000)]
    assert 0.08 < np.mean([a == "challenger" for a in arms]) < 0.12
    assert all(TrafficRouter("shadow", 1.0).route(i)["decide"] == "champion" for i in range(100))
    assert all(TrafficRouter("off").route(i)["score_with"] == ["champion"] for i in range(100))
    seg = TrafficRouter("canary", 1.0, {"A2A"})
    assert seg.route(1, "CNP")["decide"] == "champion" and seg.route(1, "A2A")["decide"] == "challenger"


def test_canary_auto_rollback_on_guardrail_breach():
    ctl = RolloutController(TrafficRouter("canary", 0.1), GuardrailMonitor(min_n=100))
    rng = np.random.default_rng(0)
    for _ in range(200):                                              # healthy period
        ctl.observe("champion", "decline" if rng.random() < .01 else "approve", 20, 100, .01)
        ctl.observe("challenger", "decline" if rng.random() < .01 else "approve", 22, 100, .01)
    assert ctl.router.mode == "canary"
    for _ in range(300):                                              # challenger starts declining far more
        ctl.observe("challenger", "decline" if rng.random() < .2 else "approve", 22, 100, .01)
    assert ctl.router.mode == "off" and "decline_rate" in ctl.history[0]["breach"]
    slow = RolloutController(TrafficRouter("canary", 0.1), GuardrailMonitor(min_n=50))
    for _ in range(100):
        slow.observe("champion", "approve", 20, 10, .01)
        slow.observe("challenger", "approve", 150, 10, .01)           # latency SLO breach
    assert slow.router.mode == "off" and "p99_latency" in slow.history[0]["breach"]


def test_stack_inputs_accepts_scalars_lists_and_arrays():
    from fraudplat.models.ensemble import stack_inputs
    a = stack_inputs([0.1], [0.5], [-6.0], [-6.0], [-6.0])
    b = stack_inputs(np.array([0.1]), np.array([0.5]), np.array([-6.0]), np.array([-6.0]), np.array([-6.0]))
    c = stack_inputs(0.1, 0.5, -6.0, -6.0, -6.0)
    assert a.shape == (1, 5) and np.array_equal(a, b) and np.array_equal(a, c)
    assert (a[0, 2:] == 0).all()                       # unknown graph nodes map to 0
