"""Reactive (HPA-like) baseline (Part F).

desired = ceil(ready * cpu / target_util) with a tolerance band and a scale-down stabilisation window
(the maximum desired value over the last K intervals), as in the Kubernetes HPA algorithm. This is a
simulator approximation of HPA, NOT a full HPA implementation; use a real HPA for the Kubernetes stage.
"""
from __future__ import annotations

from collections import deque

import numpy as np

from ..controller.policy import BasePolicy, delta_to_action


class ReactivePolicy(BasePolicy):
    def __init__(self, target_util: float = 0.7, tolerance: float = 0.1,
                 scale_down_stabilization_steps: int = 6, name: str = "reactive"):
        self.target, self.tol, self.K, self.name = target_util, tolerance, int(scale_down_stabilization_steps), name
        self.hist: deque = deque(maxlen=max(self.K, 1))

    def reset(self, state: dict) -> None:
        self.hist.clear()

    def act(self, state: dict, obs=None) -> int:
        ready, total = state["ready"], state["total"]
        ratio = state["cpu"] / self.target
        desired = total if abs(ratio - 1.0) <= self.tol else int(np.ceil(ready * ratio))
        desired = int(np.clip(desired, state["min_replicas"], state["max_replicas"]))
        self.hist.append(desired)
        if desired > total:
            return delta_to_action(desired - total, state["deltas"])
        stabilised = max(self.hist)              # scale down only when ALL of the last K desires were lower
        if len(self.hist) >= self.K and stabilised < total:
            return delta_to_action(stabilised - total, state["deltas"])
        return delta_to_action(0, state["deltas"])
