"""Controller safety shield (Part E).

Reward penalties alone do not guarantee safety. Every action - from PPO or any baseline - passes
through this shield *before* reaching the simulator:
  1. invalid action ids are replaced (no-op) and flagged;
  2. replica bounds are enforced on ready+pending (min/max);
  3. scale-down is vetoed when observed demand would exceed `scale_down_max_util` of the remaining ready
     capacity (capacity feasibility);
  4. optional emergency fallback: force a scale-up when observed utilisation is above `emergency_util`.
The simulator additionally enforces the hard bounds itself (defence in depth).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class SafetyConfig:
    enabled: bool = True
    min_replicas: int = 2
    max_replicas: int = 24
    scale_down_max_util: float = 0.85
    emergency_scale_up: bool = False
    emergency_util: float = 1.0
    emergency_delta: int = 1
    invalid_action: str = "noop"           # "noop" | "raise"
    capacity_per_replica: float = 100.0

    @classmethod
    def from_config(cls, cfg: dict, capacity_per_replica: float) -> "SafetyConfig":
        s, c = cfg["safety"], cfg["cluster"]
        return cls(enabled=bool(s["enabled"]), min_replicas=int(c["min_replicas"]), max_replicas=int(c["max_replicas"]),
                   scale_down_max_util=float(s["scale_down_max_util"]), emergency_scale_up=bool(s["emergency_scale_up"]),
                   emergency_util=float(s["emergency_util"]), invalid_action=str(s["invalid_action"]),
                   capacity_per_replica=float(capacity_per_replica))


class SafetyShield:
    def __init__(self, cfg: SafetyConfig, deltas: list[int]):
        self.cfg = cfg
        self.deltas = list(deltas)

    def apply(self, action, ready: int, pending: int, demand_now: float) -> tuple[int, dict]:
        """Return (safe replica delta, info). `demand_now` is the demand observed at decision time."""
        c = self.cfg
        info = {"intervened": False, "reason": "", "requested_delta": None}
        try:
            a = int(action)
            valid = 0 <= a < len(self.deltas)
        except (TypeError, ValueError):
            a, valid = -1, False
        if not valid:
            if c.invalid_action == "raise":
                raise ValueError(f"invalid action {action!r}")
            info.update(intervened=True, reason="invalid_action")
            return 0, info
        delta = int(self.deltas[a])
        info["requested_delta"] = delta
        if not c.enabled:
            return delta, info
        total = ready + pending
        if delta > 0 and total + delta > c.max_replicas:
            new = max(0, c.max_replicas - total)
            info.update(intervened=True, reason="max_replicas")
            delta = new
        if delta < 0:
            if total + delta < c.min_replicas:
                delta = min(0, c.min_replicas - total)
                info.update(intervened=True, reason="min_replicas")
            if delta < 0:
                # capacity feasibility: removing ready replicas must keep utilisation under the limit.
                # (cancelling still-pending replicas does not reduce serving capacity)
                removal_from_ready = max(0, -delta - pending)
                remaining = ready - removal_from_ready
                if removal_from_ready > 0 and demand_now > c.scale_down_max_util * remaining * c.capacity_per_replica:
                    delta = -min(-delta, pending)
                    info.update(intervened=True, reason="scale_down_capacity")
        if c.emergency_scale_up and delta <= 0 and ready > 0:
            util = demand_now / (ready * c.capacity_per_replica)
            if util >= c.emergency_util and total < c.max_replicas and pending == 0:
                delta = min(c.emergency_delta, c.max_replicas - total)
                info.update(intervened=True, reason="emergency_scale_up")
        return int(delta), info
