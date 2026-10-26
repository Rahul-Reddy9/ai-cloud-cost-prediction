"""Cluster resource simulator (Part D).

Models, per interval: workload demand, ready replicas and capacity per replica (limited by both CPU
and memory), requested-but-not-ready (pending) replicas with provisioning delay, cost, latency
degradation when demand approaches/exceeds capacity, SLO violations and scaling actions.

Timing convention: a scale request is made at a decision boundary between interval t and t+1.
A replica requested with delay d >= 1 is billed immediately but only serves after it has been
pending for d full intervals; d = 0 means it serves the very next interval.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .cost_model import PriceModel


@dataclass
class SimulatorConfig:
    min_replicas: int = 2
    max_replicas: int = 24
    initial_replicas: int = 6
    cpu_cores_per_replica: float = 2.0
    cpu_seconds_per_request: float = 0.02
    memory_gb_per_replica: float = 4.0
    memory_usable_fraction: float = 0.9
    memory_base_gb: float = 0.8
    memory_gb_per_rps: float = 0.02
    delay_steps: int = 3
    jitter_steps: int = 0
    max_delay_steps: int = 6
    bill_pending: bool = True
    base_latency_ms: float = 40.0
    rho_cap: float = 0.95
    overload_slope: float = 5.0
    timeout_ms: float = 5000.0
    slo_latency_ms: float = 300.0
    metric_noise: float = 0.01
    interval_minutes: float = 5.0

    @property
    def cpu_capacity_rps(self) -> float:
        return self.cpu_cores_per_replica / self.cpu_seconds_per_request

    @property
    def memory_capacity_rps(self) -> float:
        usable = self.memory_gb_per_replica * self.memory_usable_fraction - self.memory_base_gb
        return max(usable, 0.0) / self.memory_gb_per_rps

    @property
    def capacity_per_replica(self) -> float:
        """Effective req/s per replica = min(CPU-bound, memory-bound)."""
        return float(min(self.cpu_capacity_rps, self.memory_capacity_rps))

    @classmethod
    def from_config(cls, cfg: dict) -> "SimulatorConfig":
        c, p, l = cfg["cluster"], cfg["provisioning"], cfg["latency"]
        return cls(
            min_replicas=int(c["min_replicas"]), max_replicas=int(c["max_replicas"]),
            initial_replicas=int(c["initial_replicas"]),
            cpu_cores_per_replica=float(c["cpu_cores_per_replica"]),
            cpu_seconds_per_request=float(c["cpu_seconds_per_request"]),
            memory_gb_per_replica=float(c["memory_gb_per_replica"]),
            memory_usable_fraction=float(c["memory_usable_fraction"]),
            memory_base_gb=float(c["memory_base_gb"]), memory_gb_per_rps=float(c["memory_gb_per_rps"]),
            delay_steps=int(p["delay_steps"]), jitter_steps=int(p["jitter_steps"]),
            max_delay_steps=int(p["max_delay_steps"]), bill_pending=bool(p["bill_pending"]),
            base_latency_ms=float(l["base_ms"]), rho_cap=float(l["rho_cap"]),
            overload_slope=float(l["overload_slope"]), timeout_ms=float(l["timeout_ms"]),
            slo_latency_ms=float(cfg["slo"]["latency_ms"]),
            metric_noise=float(cfg["noise"]["metric_noise"]),
            interval_minutes=float(cfg["price"]["interval_minutes"]),
        )


def latency_model(rho: float, cfg: SimulatorConfig) -> float:
    """Documented, SYNTHETIC latency approximation (not a measurement).

    Below rho_cap: base / (1 - rho)  (queueing-like growth).  Above: continues linearly from the value
    at rho_cap with slope `overload_slope`, capped at the timeout.
    """
    rho = max(float(rho), 0.0)
    at_cap = cfg.base_latency_ms / (1.0 - cfg.rho_cap)
    if rho < cfg.rho_cap:
        lat = cfg.base_latency_ms / (1.0 - rho)
    else:
        lat = at_cap * (1.0 + cfg.overload_slope * (rho - cfg.rho_cap))
    return float(min(lat, cfg.timeout_ms))


class ClusterSimulator:
    """Discrete-time replica simulator with provisioning delays."""

    def __init__(self, config: SimulatorConfig, price: PriceModel, seed: int = 0):
        self.cfg = config
        self.price = price
        self.rng = np.random.default_rng(seed)
        self.ready = config.initial_replicas
        self.pending = np.zeros(config.max_delay_steps + 1, dtype=int)  # index r = intervals left (1..D)
        self.t = 0

    # ----- state ---------------------------------------------------------------------------
    def reset(self, initial_replicas: int | None = None, seed: int | None = None) -> None:
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        n = self.cfg.initial_replicas if initial_replicas is None else int(initial_replicas)
        self.ready = int(np.clip(n, self.cfg.min_replicas, self.cfg.max_replicas))
        self.pending[:] = 0
        self.t = 0

    @property
    def pending_total(self) -> int:
        return int(self.pending.sum())

    @property
    def total_replicas(self) -> int:
        return self.ready + self.pending_total

    @property
    def billable(self) -> int:
        return self.ready + (self.pending_total if self.cfg.bill_pending else 0)

    def pending_by_eta(self) -> np.ndarray:
        """Replicas that become ready in 1..max_delay intervals (index 0 -> eta 1)."""
        return self.pending[1:].copy()

    # ----- scaling ---------------------------------------------------------------------------
    def request_scaling(self, delta: int) -> int:
        """Apply a scaling request, enforcing hard replica bounds. Returns the delta actually applied."""
        cfg = self.cfg
        delta = int(delta)
        if delta > 0:
            add = max(0, min(delta, cfg.max_replicas - self.total_replicas))
            if add == 0:
                return 0
            d = cfg.delay_steps
            if d > 0 and cfg.jitter_steps > 0:
                d = int(np.clip(d + self.rng.integers(-cfg.jitter_steps, cfg.jitter_steps + 1),
                                1, cfg.max_delay_steps))
            d = min(d, cfg.max_delay_steps)
            if d <= 0:
                self.ready += add
            else:
                self.pending[d] += add
            return add
        if delta < 0:
            remaining = -delta
            removed = 0
            for r in range(cfg.max_delay_steps, 0, -1):          # cancel not-yet-ready replicas first
                take = min(remaining, int(self.pending[r]))
                self.pending[r] -= take
                remaining -= take
                removed += take
                if remaining == 0:
                    break
            if remaining > 0:                                    # then terminate ready replicas
                can = max(0, self.ready - cfg.min_replicas)
                take = min(remaining, can)
                self.ready -= take
                removed += take
            return -removed
        return 0

    # ----- metrics for an interval -------------------------------------------------------------
    def interval_metrics(self, demand: float, ready: int | None = None) -> dict:
        cfg = self.cfg
        ready = self.ready if ready is None else ready
        cap = max(ready, 1) * cfg.capacity_per_replica
        rho = demand / cap
        served = min(demand, cap)
        dropped = max(demand - cap, 0.0)
        per_replica = served / max(ready, 1)
        cpu = per_replica / cfg.cpu_capacity_rps
        mem = (cfg.memory_base_gb + cfg.memory_gb_per_rps * per_replica) / cfg.memory_gb_per_replica
        if cfg.metric_noise > 0:
            cpu += self.rng.normal(0, cfg.metric_noise)
            mem += self.rng.normal(0, cfg.metric_noise)
        latency = latency_model(rho, cfg)
        dropped_frac = dropped / demand if demand > 0 else 0.0
        violated = bool(latency > cfg.slo_latency_ms or dropped > 1e-9)
        return {
            "capacity": cap, "rho": float(rho), "served": float(served), "dropped": float(dropped),
            "dropped_fraction": float(dropped_frac),
            "cpu_utilization": float(np.clip(cpu, 0.0, 1.0)),
            "memory_utilization": float(np.clip(mem, 0.0, 1.0)),
            "latency_ms": latency, "violated": violated, "under_provisioned": bool(rho > 1.0),
        }

    def step(self, demand: float) -> dict:
        """Serve one interval with the current ready replicas, then advance provisioning clocks."""
        ready_serving = self.ready
        pending_after_request = self.pending_total
        billable = self.billable
        rec = self.interval_metrics(float(demand), ready_serving)
        rec.update({
            "demand": float(demand), "ready": ready_serving, "pending": pending_after_request,
            "billable": billable, "cost": self.price.interval_cost(billable),
        })
        # advance provisioning clocks: replicas with 1 interval left become ready
        arrivals = int(self.pending[1])
        self.pending[1:-1] = self.pending[2:]
        self.pending[-1] = 0
        self.ready += arrivals
        self.t += 1
        return rec
