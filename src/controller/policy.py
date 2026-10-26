"""Policy interface shared by PPO and the rule-based baselines (Part E/F)."""
from __future__ import annotations

from pathlib import Path

import numpy as np


def delta_to_action(desired_delta: float, deltas: list[int]) -> int:
    """Map a desired replica change onto the discrete action set without overshooting its magnitude."""
    if desired_delta >= 0:
        cands = [(d, i) for i, d in enumerate(deltas) if 0 <= d <= desired_delta]
        return max(cands)[1] if cands else deltas.index(0)
    cands = [(d, i) for i, d in enumerate(deltas) if desired_delta <= d <= 0]
    return min(cands)[1] if cands else deltas.index(0)


class BasePolicy:
    """act(state, obs) -> action index.  `state` is the rich decision-time dict from the environment."""
    name = "base"

    def reset(self, state: dict) -> None:
        pass

    def initial_replicas(self, state: dict) -> int | None:
        return None

    def act(self, state: dict, obs: np.ndarray | None = None) -> int:
        raise NotImplementedError


class PPOPolicy(BasePolicy):
    """Wraps a trained Stable-Baselines3 PPO model.

    Evaluation SAMPLES actions from the learned policy with a fixed RNG seed (reproducible).  Argmax
    evaluation is available (deterministic=True) but was found to collapse to the no-op action whenever
    the learned action probabilities are split (e.g. 0.56 keep / 0.40 +1), so it is not the default.
    """

    def __init__(self, model_path: str | Path, name: str = "ppo", deterministic: bool = False, sample_seed: int = 0):
        from stable_baselines3 import PPO
        self.model = PPO.load(str(model_path), device="cpu")
        self.name, self.deterministic, self.sample_seed = name, deterministic, int(sample_seed)

    def reset(self, state: dict) -> None:
        import torch
        torch.manual_seed(self.sample_seed)

    def act(self, state: dict, obs: np.ndarray | None = None) -> int:
        a, _ = self.model.predict(obs, deterministic=self.deterministic)
        return int(a)
