"""Cost model and reward terms.

Simulated cost is  C_total = sum_t n_t * p * dt  with n_t billable replicas in interval t,
p the price per replica-hour and dt the interval length in hours.  It is a *simulated* quantity
and must never be presented as a cloud invoice.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np


@dataclass(frozen=True)
class PriceModel:
    price_per_replica_hour: float = 0.05
    interval_minutes: float = 5.0
    currency: str = "USD"

    @property
    def dt_hours(self) -> float:
        return self.interval_minutes / 60.0

    def interval_cost(self, billable_replicas: float) -> float:
        return float(billable_replicas) * self.price_per_replica_hour * self.dt_hours

    def total_cost(self, billable_series: Iterable[float]) -> float:
        arr = np.asarray(list(billable_series), dtype=float)
        return float(arr.sum() * self.price_per_replica_hour * self.dt_hours)

    def max_interval_cost(self, max_replicas: int) -> float:
        return self.interval_cost(max_replicas)

    @classmethod
    def from_config(cls, cfg: dict) -> "PriceModel":
        p = cfg["price"] if "price" in cfg else cfg
        return cls(float(p["price_per_replica_hour"]), float(p["interval_minutes"]),
                   p.get("currency", "USD"))


def cost_reduction_pct(baseline_cost: float, proposed_cost: float) -> float:
    """100 * (C_baseline - C_proposed) / C_baseline ; positive => proposed is cheaper."""
    if baseline_cost == 0:
        return float("nan")
    return 100.0 * (baseline_cost - proposed_cost) / baseline_cost


@dataclass(frozen=True)
class RewardWeights:
    alpha: float = 1.0      # cost
    beta: float = 0.2       # latency
    gamma: float = 1.5      # SLO violation
    delta: float = 0.05     # scaling churn
    latency_cap_ratio: float = 3.0

    @classmethod
    def from_config(cls, cfg: dict) -> "RewardWeights":
        r = cfg["reward"] if "reward" in cfg else cfg
        return cls(**{k: float(v) for k, v in r.items()})


def compute_reward_terms(cost: float, max_cost: float, latency_ms: float, slo_ms: float,
                         violated: bool, dropped_fraction: float, applied_change: float,
                         max_step: float, w: RewardWeights) -> dict:
    """Return the normalised penalty terms and reward r = -(a*C + b*L + g*V + d*H).

    C: cost / maximum possible interval cost.      L: min(latency/SLO, cap).
    V: 1[SLO violated] + dropped request fraction. H: |replica change| / max step.
    All four terms are >= 0, so reward is monotonically non-increasing in each.
    """
    C = float(cost) / max(float(max_cost), 1e-12)
    L = float(min(max(latency_ms, 0.0) / max(slo_ms, 1e-9), w.latency_cap_ratio))
    V = float(bool(violated)) + float(max(dropped_fraction, 0.0))
    H = abs(float(applied_change)) / max(float(max_step), 1e-9)
    reward = -(w.alpha * C + w.beta * L + w.gamma * V + w.delta * H)
    return {"C": C, "L": L, "V": V, "H": H, "reward": float(reward)}
