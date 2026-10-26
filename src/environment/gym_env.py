"""Gymnasium environment: one step = one resource-management interval (Part E).

State s_t = [u_t, m_t, demand_t, hour sin/cos, ready replicas,
             d_hat (point forecasts at several horizons),
             q (forecast uncertainty: upper-lower at the same horizons)   [if use_uncertainty],
             p (pending replicas by remaining delay)                      [if use_pending],
             c (cost state), z (anomaly score)                            [if anomaly != none]]
Everything in the state is available at decision time t (forecasts at origin t use data <= t; anomaly
scores are computed from data <= t).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from ..controller.safety import SafetyConfig, SafetyShield
from .cost_model import PriceModel, RewardWeights, compute_reward_terms
from .simulator import ClusterSimulator, SimulatorConfig
from .workload import WorkloadTrace


@dataclass
class EnvConfig:
    sim: SimulatorConfig
    price: PriceModel
    reward: RewardWeights
    safety: SafetyConfig
    deltas: list[int] = field(default_factory=lambda: [-1, 0, 1, 3])
    forecast_steps: list[int] = field(default_factory=lambda: [1, 3, 6, 12])
    episode_length: int = 288
    initial_util: float = 0.65
    use_forecast: bool = True
    use_uncertainty: bool = False
    use_pending: bool = False
    anomaly: str = "none"                 # none | raw | residual

    @classmethod
    def from_config(cls, sim_cfg: dict, variant: dict | None = None, delay_steps: int | None = None) -> "EnvConfig":
        sim = SimulatorConfig.from_config(sim_cfg)
        if delay_steps is not None:
            sim.delay_steps = int(delay_steps)
        variant = variant or {}
        return cls(
            sim=sim, price=PriceModel.from_config(sim_cfg), reward=RewardWeights.from_config(sim_cfg),
            safety=SafetyConfig.from_config(sim_cfg, sim.capacity_per_replica),
            deltas=list(sim_cfg["actions"]["deltas"]), forecast_steps=list(sim_cfg["episode"]["forecast_steps"]),
            episode_length=int(sim_cfg["episode"]["length_steps"]), initial_util=float(sim_cfg["episode"]["initial_util"]),
            use_uncertainty=bool(variant.get("use_uncertainty", False)),
            use_pending=bool(variant.get("use_pending", False)), anomaly=str(variant.get("anomaly", "none")))


def feature_names(cfg: EnvConfig) -> list[str]:
    names = ["cpu_util", "mem_util", "demand_obs", "hour_sin", "hour_cos", "ready_frac"]
    names += [f"fc_median_h{h}" for h in cfg.forecast_steps]
    names += [f"fc_rho_median_h{h}" for h in cfg.forecast_steps]
    if cfg.use_uncertainty:
        names += [f"fc_width_h{h}" for h in cfg.forecast_steps]
        names += [f"fc_rho_upper_h{h}" for h in cfg.forecast_steps]
    if cfg.use_pending:
        names += [f"pending_eta{k}" for k in range(1, cfg.sim.max_delay_steps + 1)]
    names += ["cost_state"]
    if cfg.anomaly != "none":
        names += ["anomaly_z"]
    return names


class CloudScalingEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, traces: list[WorkloadTrace], config: EnvConfig, mode: str = "train", seed: int = 0):
        super().__init__()
        assert traces, "need at least one trace"
        self.traces, self.cfg, self.mode = traces, config, mode
        self.sim = ClusterSimulator(config.sim, config.price, seed=seed)
        self.shield = SafetyShield(config.safety, config.deltas)
        self.action_space = spaces.Discrete(len(config.deltas))
        n = len(feature_names(config))
        self.observation_space = spaces.Box(-5.0, 5.0, shape=(n,), dtype=np.float32)
        self.cap_norm = config.sim.capacity_per_replica * config.sim.max_replicas
        self.demand_scale = 0.5 * self.cap_norm      # typical demand maps to ~0.2-0.8
        self.width_scale = 0.1 * self.cap_norm       # typical 10-90% width maps to ~0.1-1
        self.max_step = max(abs(d) for d in config.deltas)
        self.max_cost = config.price.max_interval_cost(config.sim.max_replicas)
        self._rng = np.random.default_rng(seed)
        self._eval_cursor = 0
        self.trace: WorkloadTrace | None = None
        self.i = 0

    # ------------------------------------------------------------------------------------------
    def _forecast_arrays(self, i: int):
        """(median, upper, width) at the configured horizons; persistence when no forecasts exist."""
        tr = self.trace
        steps = self.cfg.forecast_steps
        if tr.forecasts is None:
            d = np.full(len(steps), tr.demand[i])
            return d, d, np.zeros(len(steps))
        f = tr.forecasts[i]
        idx = [h - 1 for h in steps]
        return f[idx, 1], f[idx, 2], f[idx, 2] - f[idx, 0]

    def _capacity_at(self, h: int) -> float:
        """Replicas able to serve the interval h steps ahead: ready (+ pending that arrive in time)."""
        n = self.sim.ready
        if self.cfg.use_pending:
            pend = self.sim.pending_by_eta()           # index k-1 -> replicas ready after k more intervals
            n += int(pend[: max(h - 1, 0)].sum())
        return max(n, 1) * self.cfg.sim.capacity_per_replica

    def _observe(self, metrics: dict) -> np.ndarray:
        tr, c, i = self.trace, self.cfg, self.i
        o = [metrics["cpu_utilization"], metrics["memory_utilization"],
             tr.demand[i] / self.demand_scale, tr.hour_sin[i], tr.hour_cos[i], self.sim.ready / c.sim.max_replicas]
        med, up, width = self._forecast_arrays(i)
        caps = np.array([self._capacity_at(h) for h in c.forecast_steps])
        o += list(np.clip(med / self.demand_scale, 0, 3))
        o += list(np.clip(med / caps, 0, 3))
        if c.use_uncertainty:
            o += list(np.clip(width / self.width_scale, 0, 3))
            o += list(np.clip(up / caps, 0, 3))
        if c.use_pending:
            o += list(self.sim.pending_by_eta() / c.sim.max_replicas)
        # cost state: running cost fraction. Without pending information only READY capacity is visible,
        # so the cost signal cannot leak pending replicas.
        visible = self.sim.billable if c.use_pending else self.sim.ready
        self._cost_acc += visible / c.sim.max_replicas
        o += [self._cost_acc / max(self._t + 1, 1)]
        if c.anomaly != "none":
            z = tr.z_raw if c.anomaly == "raw" else tr.z_resid
            o += [0.0 if z is None or np.isnan(z[i]) else float(z[i])]
        return np.clip(np.asarray(o, dtype=np.float32), -5.0, 5.0)

    def _initial_replicas(self, i: int) -> int:
        cap = self.cfg.sim.capacity_per_replica
        n = int(np.ceil(self.trace.demand[i] / (cap * self.cfg.initial_util)))
        return int(np.clip(n, self.cfg.sim.min_replicas, self.cfg.sim.max_replicas))

    # ------------------------------------------------------------------------------------------
    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        options = options or {}
        if "trace_index" in options:
            k = int(options["trace_index"])
        elif self.mode == "train":
            k = int(self._rng.integers(len(self.traces)))
        else:
            k = self._eval_cursor % len(self.traces)
            self._eval_cursor += 1
        self.trace = self.traces[k]
        lo, hi = self.trace.first_valid_origin(), self.trace.last_valid_origin()
        if self.mode == "train":
            L = min(self.cfg.episode_length, max(hi - lo, 1))
            self.i = int(self._rng.integers(lo, max(lo + 1, hi - L + 1)))
            self.end = min(self.i + L, hi)
        else:
            self.i, self.end = lo, hi
        self._t = 0
        self._cost_acc = 0.0
        n0 = options.get("initial_replicas") or self._initial_replicas(self.i)
        self.sim.reset(n0, seed=int(self._rng.integers(1 << 30)))
        m = self.sim.interval_metrics(self.trace.demand[self.i])
        self._last_metrics = m
        self.start_index = self.i
        return self._observe(m), {"trace": self.trace.name, "start_index": self.i, "end_index": self.end}

    def decision_state(self) -> dict:
        """Rich dict view of the current decision-time state for rule-based policies."""
        tr, i = self.trace, self.i
        fc = None if tr.forecasts is None else tr.forecasts[i]
        m = self._last_metrics
        return {
            "i": i, "t": self._t, "demand": float(tr.demand[i]), "cpu": m["cpu_utilization"], "mem": m["memory_utilization"],
            "ready": self.sim.ready, "pending": self.sim.pending_total, "pending_by_eta": self.sim.pending_by_eta(),
            "total": self.sim.total_replicas, "forecast": fc, "capacity_per_replica": self.cfg.sim.capacity_per_replica,
            "min_replicas": self.cfg.sim.min_replicas, "max_replicas": self.cfg.sim.max_replicas,
            "z_raw": None if tr.z_raw is None else float(tr.z_raw[i]),
            "z_resid": None if tr.z_resid is None else float(tr.z_resid[i]),
            "deltas": self.cfg.deltas, "delay_steps": self.cfg.sim.delay_steps,
        }

    def step(self, action):
        tr, c = self.trace, self.cfg
        demand_now = float(tr.demand[self.i])
        delta, sinfo = self.shield.apply(action, self.sim.ready, self.sim.pending_total, demand_now)
        applied = self.sim.request_scaling(delta)
        rec = self.sim.step(float(tr.demand[self.i + 1]))
        self.i += 1
        self._t += 1
        terms = compute_reward_terms(rec["cost"], self.max_cost, rec["latency_ms"], c.sim.slo_latency_ms,
                                     rec["violated"], rec["dropped_fraction"], applied, self.max_step, c.reward)
        self._last_metrics = rec
        obs = self._observe(rec)
        terminated = False
        truncated = self.i >= self.end
        info = dict(rec)
        info.update({"applied_delta": applied, "requested_delta": sinfo["requested_delta"],
                     "safety_intervened": bool(sinfo["intervened"]), "safety_reason": sinfo["reason"],
                     "reward": terms["reward"], "terms": terms})
        return obs, terms["reward"], terminated, truncated, info
