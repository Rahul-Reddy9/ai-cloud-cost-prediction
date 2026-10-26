"""Predictive baseline (Part F): size capacity from TFT forecasts of the demand that will be present when
newly requested replicas become ready (horizon = provisioning delay + 1).

use_quantile="median" -> point-forecast scaling.  "upper" -> scale for the TFT upper quantile
(a simple, non-learned way to use forecast uncertainty).
"""
from __future__ import annotations

from collections import deque

import numpy as np

from ..controller.policy import BasePolicy, delta_to_action


class PredictivePolicy(BasePolicy):
    def __init__(self, target_util: float = 0.7, scale_down_stabilization_steps: int = 3,
                 use_quantile: str = "median", window: int = 3, name: str | None = None):
        assert use_quantile in ("median", "upper")
        self.target, self.K, self.q, self.window = target_util, int(scale_down_stabilization_steps), use_quantile, int(window)
        self.name = name or ("predictive" if use_quantile == "median" else "predictive_q")
        self.hist: deque = deque(maxlen=max(self.K, 1))

    def reset(self, state: dict) -> None:
        self.hist.clear()

    def act(self, state: dict, obs=None) -> int:
        cap = state["capacity_per_replica"] * self.target
        fc = state["forecast"]
        need_now = state["demand"] / cap
        if fc is None or np.isnan(fc[:, 1]).any():
            need = need_now
        else:
            h0 = min(state["delay_steps"] + 1, fc.shape[0])              # first step served by new replicas
            h1 = min(h0 + self.window, fc.shape[0])
            col = 1 if self.q == "median" else 2
            need = max(need_now, float(np.max(fc[h0 - 1:h1, col])) / cap)
        desired = int(np.clip(np.ceil(need), state["min_replicas"], state["max_replicas"]))
        self.hist.append(desired)
        total = state["total"]
        if desired > total:
            return delta_to_action(desired - total, state["deltas"])
        if len(self.hist) >= self.K and max(self.hist) < total:
            return delta_to_action(max(self.hist) - total, state["deltas"])
        return delta_to_action(0, state["deltas"])
