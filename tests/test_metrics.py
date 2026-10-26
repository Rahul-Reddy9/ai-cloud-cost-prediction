import numpy as np
import pandas as pd
import pytest

from src.baselines.fixed_policy import FixedPolicy, size_fixed
from src.baselines.predictive_policy import PredictivePolicy
from src.baselines.reactive_policy import ReactivePolicy
from src.controller.policy import delta_to_action
from src.evaluation import metrics as M


def test_forecast_metrics_known_values():
    y, p = np.array([1.0, 2.0, 3.0]), np.array([2.0, 2.0, 5.0])
    assert M.mae(y, p) == pytest.approx(1.0)
    assert M.rmse(y, p) == pytest.approx(np.sqrt(5 / 3))
    assert M.pinball_loss(y, p, 0.5) == pytest.approx(0.5 * M.mae(y, p))
    assert M.pinball_loss([10.0], [8.0], 0.9) == pytest.approx(0.9 * 2)     # under-prediction
    assert M.pinball_loss([8.0], [10.0], 0.9) == pytest.approx(0.1 * 2)     # over-prediction
    assert M.interval_coverage([1, 5, 9], [0, 0, 0], [2, 6, 7]) == pytest.approx(2 / 3)
    assert M.interval_width([0, 1], [2, 5]) == pytest.approx(3.0)


def test_detection_metrics_known_values():
    r = M.detection_metrics([1, 1, 0, 0, 0, 1], [1, 0, 1, 0, 0, 1])
    assert (r["tp"], r["fp"], r["fn"], r["tn"]) == (2, 1, 1, 2)
    assert r["precision"] == pytest.approx(2 / 3) and r["recall"] == pytest.approx(2 / 3)
    assert r["f1"] == pytest.approx(2 / 3) and r["fpr"] == pytest.approx(1 / 3)
    assert M.detection_metrics([0, 0], [0, 0])["f1"] == 0.0


def steps(**over):
    n = 10
    d = {"demand": np.full(n, 100.0), "cost": np.full(n, 0.1), "latency_ms": np.full(n, 50.0),
         "violated": np.zeros(n, bool), "dropped": np.zeros(n), "ready": np.full(n, 5), "billable": np.full(n, 5),
         "applied_delta": np.zeros(n), "under_provisioned": np.zeros(n, bool), "reward": np.full(n, -0.1)}
    d.update(over)
    return pd.DataFrame(d)


def test_episode_metrics_basics():
    ad = np.zeros(10); ad[[2, 5]] = [1, -1]
    viol = np.zeros(10, bool); viol[[3, 4]] = True
    m = M.episode_metrics(steps(applied_delta=ad, violated=viol, under_provisioned=viol), 5.0)
    assert m["total_cost"] == pytest.approx(1.0)
    assert m["scaling_events"] == 2 and m["replicas_changed"] == 2
    assert m["slo_violation_rate_intervals"] == pytest.approx(0.2)
    assert m["underprovisioning_minutes"] == pytest.approx(10.0)
    assert m["scaling_events_per_day"] == pytest.approx(2 / (10 * 5 / 1440))


def test_tail_latency_is_request_weighted():
    lat = np.r_[np.full(9, 50.0), 1000.0]
    heavy = M.episode_metrics(steps(latency_ms=lat, demand=np.r_[np.full(9, 1.0), 1000.0]))
    light = M.episode_metrics(steps(latency_ms=lat, demand=np.r_[np.full(9, 1000.0), 1.0]))
    assert heavy["p95_latency_ms"] > light["p95_latency_ms"]


def test_summarise_seeds_reports_variability():
    s = M.summarise_seeds([{"a": 1.0}, {"a": 3.0}, {"a": 5.0}])
    assert s["a"]["mean"] == 3.0 and s["a"]["std"] == pytest.approx(2.0) and s["a"]["n"] == 3


# ---------------- baselines ---------------------------------------------------------------------
def state(**kw):
    s = {"ready": 6, "pending": 0, "total": 6, "cpu": 0.5, "demand": 300.0, "forecast": None, "capacity_per_replica": 100.0,
         "min_replicas": 2, "max_replicas": 24, "deltas": [-1, 0, 1, 3], "delay_steps": 3}
    s.update(kw)
    return s


def test_fixed_policy_never_scales():
    p = FixedPolicy(8)
    assert p.initial_replicas(None) == 8 and p.act(state(cpu=1.0)) == 1


def test_fixed_sizing_uses_given_demand_quantile():
    d = np.r_[np.full(95, 100.0), np.full(5, 1000.0)]
    assert size_fixed(d, 100.0, 0.7, None, 2, 24) == 15
    assert size_fixed(d, 100.0, 0.7, 0.5, 2, 24) == 2


def test_reactive_scales_up_on_high_cpu_and_waits_to_scale_down():
    p = ReactivePolicy(target_util=0.6, scale_down_stabilization_steps=3)
    assert p.act(state(cpu=0.95)) in (2, 3)
    p.reset(None)
    assert p.act(state(cpu=0.2)) == 1                      # stabilisation window not yet filled with low desires
    assert p.act(state(cpu=0.2)) == 1
    assert p.act(state(cpu=0.2)) == 0                      # all of the last 3 desires lower -> scale down


def test_predictive_policy_scales_ahead_of_forecast_demand():
    fc = np.zeros((12, 3)); fc[:, :] = 300.0; fc[3:, :] = 1000.0     # surge arrives when new replicas would be ready
    p = PredictivePolicy(target_util=0.7)
    a = p.act(state(demand=300.0, forecast=fc))
    assert a == 3                                          # +3 (needs ceil(1000/70)=15 replicas)
    low = np.full((12, 3), 300.0)
    assert PredictivePolicy(target_util=0.7).act(state(demand=300.0, forecast=low, total=5, ready=5)) == 1


def test_predictive_upper_quantile_is_more_conservative():
    fc = np.zeros((12, 3)); fc[:, 0], fc[:, 1], fc[:, 2] = 300.0, 400.0, 900.0
    med = PredictivePolicy(use_quantile="median").act(state(forecast=fc, total=6, ready=6))
    up = PredictivePolicy(use_quantile="upper").act(state(forecast=fc, total=6, ready=6))
    assert up >= med


def test_action_mapping_never_overshoots():
    D = [-1, 0, 1, 3]
    assert D[delta_to_action(2, D)] == 1 and D[delta_to_action(5, D)] == 3 and D[delta_to_action(-4, D)] == -1
