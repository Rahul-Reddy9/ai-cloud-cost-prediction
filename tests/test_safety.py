import numpy as np
import pytest

from src.controller.safety import SafetyConfig, SafetyShield
from src.environment.gym_env import CloudScalingEnv, EnvConfig
from src.environment.workload import make_plain_trace

DELTAS = [-1, 0, 1, 3]


def shield(**kw):
    return SafetyShield(SafetyConfig(min_replicas=2, max_replicas=10, capacity_per_replica=100.0, **kw), DELTAS)


def test_invalid_actions_are_replaced_and_flagged():
    s = shield()
    for bad in (-1, 4, 99, None, "x"):
        d, info = s.apply(bad, 5, 0, 100.0)
        assert d == 0 and info["intervened"] and info["reason"] == "invalid_action"
    with pytest.raises(ValueError):
        SafetyShield(SafetyConfig(invalid_action="raise"), DELTAS).apply(7, 5, 0, 1.0)


def test_scale_up_cannot_exceed_max_replicas():
    s = shield()
    d, info = s.apply(3, 8, 0, 100.0)         # +3 would reach 11
    assert d == 2 and info["intervened"]
    d, _ = s.apply(2, 10, 0, 100.0)
    assert d == 0
    d, _ = s.apply(3, 6, 2, 100.0)            # pending counts towards the limit
    assert d == 2


def test_scale_down_cannot_go_below_min():
    s = shield()
    d, info = s.apply(0, 2, 0, 10.0)
    assert d == 0 and info["intervened"]


def test_scale_down_blocked_when_capacity_infeasible():
    s = shield(scale_down_max_util=0.85)
    d, info = s.apply(0, 5, 0, 450.0)         # 4 replicas * 100 * 0.85 = 340 < 450
    assert d == 0 and info["reason"] == "scale_down_capacity"
    d, info = s.apply(0, 5, 0, 300.0)         # feasible
    assert d == -1 and not info["intervened"]


def test_cancelling_pending_replicas_is_always_allowed():
    s = shield()
    d, info = s.apply(0, 5, 2, 480.0)         # removes a pending replica, not serving capacity
    assert d == -1


def test_emergency_scale_up_optional_fallback():
    s = shield(emergency_scale_up=True, emergency_util=1.0)
    d, info = s.apply(1, 4, 0, 450.0)         # agent says "keep" while overloaded
    assert d == 1 and info["reason"] == "emergency_scale_up"
    assert shield().apply(1, 4, 0, 450.0)[0] == 0


def test_disabled_shield_passes_actions_through():
    s = shield(enabled=False)
    assert s.apply(3, 9, 0, 1.0)[0] == 3


def test_random_actions_never_violate_limits_in_environment(cfgs):
    d = 600 + 500 * np.sin(np.arange(2000) / 40.0)
    tr = make_plain_trace("t", d)
    cfg = EnvConfig.from_config(cfgs["simulation"])
    env = CloudScalingEnv([tr], cfg, mode="train", seed=0)
    rng = np.random.default_rng(0)
    env.reset(seed=0)
    for _ in range(3000):
        a = int(rng.integers(0, 4)) if rng.random() > 0.05 else int(rng.integers(-3, 9))
        _, _, term, trunc, info = env.step(a)
        sim = env.sim
        assert cfg.sim.min_replicas <= sim.ready and sim.total_replicas <= cfg.sim.max_replicas
        if trunc:
            env.reset()


def test_observation_only_recommendation_is_bounded():
    import importlib.util
    from pathlib import Path
    from src.baselines.reactive_policy import ReactivePolicy
    spec = importlib.util.spec_from_file_location("observe_only", Path(__file__).resolve().parents[1] / "deploy" / "observe_only.py")
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
    rec = m.recommend(ReactivePolicy(), ready=11, pending=0, cpu=1.0, demand=0.0, min_r=2, max_r=12)
    assert 2 <= rec <= 12
