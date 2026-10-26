import numpy as np
import pytest

from src.environment.cost_model import PriceModel, RewardWeights, compute_reward_terms, cost_reduction_pct
from src.environment.gym_env import CloudScalingEnv, EnvConfig, feature_names
from src.environment.simulator import ClusterSimulator, SimulatorConfig, latency_model
from src.environment.workload import make_plain_trace


def make_sim(cfgs, **kw):
    sc = SimulatorConfig.from_config(cfgs["simulation"])
    for k, v in kw.items():
        setattr(sc, k, v)
    return ClusterSimulator(sc, PriceModel.from_config(cfgs["simulation"]), seed=0), sc


def test_resource_counts_stay_within_bounds(cfgs):
    sim, sc = make_sim(cfgs)
    rng = np.random.default_rng(0)
    sim.reset(6)
    for _ in range(3000):
        sim.request_scaling(int(rng.integers(-4, 6)))
        sim.step(float(rng.uniform(50, 2500)))
        assert sc.min_replicas <= sim.ready
        assert sim.total_replicas <= sc.max_replicas
        assert sim.total_replicas >= sc.min_replicas


def test_requested_capacity_stays_pending_until_ready(cfgs):
    sim, sc = make_sim(cfgs, delay_steps=3)
    sim.reset(6)
    assert sim.request_scaling(3) == 3
    ready_seen = []
    for k in range(5):
        rec = sim.step(500.0)
        ready_seen.append(rec["ready"])
        if k < 3:
            assert rec["pending"] == 3
    assert ready_seen == [6, 6, 6, 9, 9]          # usable only after 3 intervals


def test_zero_delay_is_immediately_usable(cfgs):
    sim, _ = make_sim(cfgs, delay_steps=0)
    sim.reset(4); sim.request_scaling(2)
    assert sim.step(300.0)["ready"] == 6


def test_pending_replicas_are_cancelled_before_ready_ones_removed(cfgs):
    sim, _ = make_sim(cfgs, delay_steps=3)
    sim.reset(6); sim.request_scaling(3)
    assert sim.request_scaling(-2) == -2
    assert sim.pending_total == 1 and sim.ready == 6


def test_cost_calculation_known_inputs(cfgs):
    pm = PriceModel(price_per_replica_hour=0.06, interval_minutes=5)
    assert pm.interval_cost(10) == pytest.approx(10 * 0.06 * 5 / 60)
    assert pm.total_cost([10, 10, 20]) == pytest.approx(40 * 0.06 / 12)
    sim, _ = make_sim(cfgs)
    sim.reset(6)
    sim.request_scaling(2)
    rec = sim.step(300.0)                         # 6 ready + 2 pending billed
    assert rec["billable"] == 8 and rec["cost"] == pytest.approx(8 * 0.05 / 12)
    assert cost_reduction_pct(100.0, 80.0) == pytest.approx(20.0)
    assert cost_reduction_pct(100.0, 120.0) == pytest.approx(-20.0)


def test_capacity_limited_by_cpu_and_memory(cfgs):
    _, sc = make_sim(cfgs)
    assert sc.capacity_per_replica == min(sc.cpu_capacity_rps, sc.memory_capacity_rps)
    sim, sc2 = make_sim(cfgs, memory_gb_per_replica=2.0)    # memory becomes the binding limit
    assert sc2.capacity_per_replica == sc2.memory_capacity_rps < sc2.cpu_capacity_rps


def test_latency_increases_with_load_and_stays_finite(cfgs):
    _, sc = make_sim(cfgs)
    rhos = np.linspace(0, 3, 60)
    lat = [latency_model(r, sc) for r in rhos]
    assert all(np.isfinite(lat)) and all(b >= a for a, b in zip(lat, lat[1:]))
    assert lat[-1] <= sc.timeout_ms


def test_overload_drops_requests_and_violates_slo(cfgs):
    sim, sc = make_sim(cfgs)
    sim.reset(2)
    rec = sim.step(500.0)                         # capacity 200
    assert rec["dropped"] == pytest.approx(300.0) and rec["violated"] and rec["under_provisioned"]


def test_reward_penalties_have_intended_signs():
    w = RewardWeights()
    base = dict(cost=0.2, max_cost=1.0, latency_ms=100, slo_ms=300, violated=False, dropped_fraction=0.0,
                applied_change=0, max_step=3, w=w)
    r0 = compute_reward_terms(**base)["reward"]
    assert r0 < 0
    for k, v in dict(cost=0.5, latency_ms=250, violated=True, dropped_fraction=0.2, applied_change=3).items():
        worse = compute_reward_terms(**{**base, k: v})["reward"]
        assert worse < r0, k
    t = compute_reward_terms(**base)
    assert all(t[x] >= 0 for x in "CLVH")


# ---------------- gym environment -------------------------------------------------------------
def demand(n=800):
    h = np.arange(n) * 5 / 60
    return 600 + 300 * np.sin(2 * np.pi * h / 24)


def fc_fn(i, d, H):
    out = np.zeros((H, 3))
    for h in range(H):
        j = min(i + h + 1, len(d) - 1)
        out[h] = [d[j] * 0.9, d[j], d[j] * 1.1]
    return out


def test_gym_env_passes_sb3_check(cfgs):
    from stable_baselines3.common.env_checker import check_env
    tr = make_plain_trace("t", demand(), forecast_fn=fc_fn)
    env = CloudScalingEnv([tr], EnvConfig.from_config(cfgs["simulation"], {"use_uncertainty": True, "use_pending": True}))
    check_env(env, warn=False)


def test_observation_excludes_pending_when_disabled(cfgs):
    tr = make_plain_trace("t", demand(), forecast_fn=fc_fn)
    a = EnvConfig.from_config(cfgs["simulation"], {"use_pending": False})
    b = EnvConfig.from_config(cfgs["simulation"], {"use_pending": True, "use_uncertainty": True})
    assert not any(n.startswith("pending") for n in feature_names(a))
    assert any(n.startswith("pending") for n in feature_names(b))
    assert not any("width" in n for n in feature_names(a))


def test_state_uses_only_information_available_at_decision_time(cfgs):
    d1 = demand(); d2 = d1.copy(); d2[300:] *= 3.0      # futures differ after index 300
    t1 = make_plain_trace("a", d1, forecast_fn=lambda i, d, H: np.tile(d1[i], (H, 3)) * 1.0)
    t2 = make_plain_trace("a", d2, forecast_fn=lambda i, d, H: np.tile(d1[i], (H, 3)) * 1.0)
    cfg = EnvConfig.from_config(cfgs["simulation"], {"use_pending": True, "use_uncertainty": True})
    outs = []
    for t in (t1, t2):
        env = CloudScalingEnv([t], cfg, mode="eval", seed=1)
        env.reset(seed=1, options={"initial_replicas": 8})
        obs = []
        for _ in range(100):                           # steps 1..100: demand[i+1] < 300 for both traces
            o, *_ = env.step(1)
            obs.append(o)
        outs.append(np.array(obs))
    assert np.allclose(outs[0], outs[1])


def test_environment_is_reproducible_with_same_seed(cfgs):
    from src.baselines.reactive_policy import ReactivePolicy
    from src.evaluation.run_experiment import evaluate_policy
    tr = make_plain_trace("t", demand(1000), forecast_fn=fc_fn)
    cfg = EnvConfig.from_config(cfgs["simulation"])
    r1 = evaluate_policy(ReactivePolicy(), [tr], cfg, seed=3)
    r2 = evaluate_policy(ReactivePolicy(), [tr], cfg, seed=3)
    assert r1["aggregate"] == r2["aggregate"]
    assert (r1["steps"].to_numpy() == r2["steps"].to_numpy()).all()


def test_run_record_stores_config_seed_and_results(cfgs, tmp_path):
    import json
    from src.evaluation.run_experiment import Context, run_and_save
    tr = make_plain_trace("t", demand(600), forecast_fn=fc_fn)
    ctx = Context(cfgs, None, {}, None, None, {"test": [tr]})
    import pandas as pd
    ctx.df = pd.DataFrame({"split": ["train"] * 10, "request_rate": np.linspace(200, 900, 10)})
    rec = run_and_save(ctx, "reactive", seed=5, split="test", metrics_dir=str(tmp_path), logs_dir=str(tmp_path))
    saved = json.load(open(tmp_path / f"{rec['run_id']}.json"))
    assert saved["seed"] == 5 and "config_hashes" in saved and "config_snapshot" in saved
    assert saved["aggregate"]["total_cost"] > 0 and (tmp_path / f"steps_{rec['run_id']}.parquet").exists()
