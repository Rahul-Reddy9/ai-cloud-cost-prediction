"""Fixed allocation baseline (Part F): constant replica count."""
from __future__ import annotations

import numpy as np

from ..controller.policy import BasePolicy, delta_to_action


def size_fixed(demand: np.ndarray, capacity_per_replica: float, target_util: float,
               quantile: float | None, min_replicas: int, max_replicas: int) -> int:
    """Replicas needed so that demand at `quantile` (or the peak when None) fits at `target_util`.
    Sized on TRAIN demand only so the baseline never sees test data."""
    q = float(np.max(demand)) if quantile is None else float(np.quantile(demand, quantile))
    return int(np.clip(np.ceil(q / (capacity_per_replica * target_util)), min_replicas, max_replicas))


class FixedPolicy(BasePolicy):
    def __init__(self, replicas: int, name: str = "fixed"):
        self.replicas, self.name = int(replicas), name

    def initial_replicas(self, state: dict) -> int:
        return self.replicas

    def act(self, state: dict, obs=None) -> int:
        return delta_to_action(0, state["deltas"])
