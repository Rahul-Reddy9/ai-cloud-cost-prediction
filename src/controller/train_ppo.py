"""PPO training (Part E).

  python -m src.controller.train_ppo --variant ppo_full --seed 0 [--timesteps N]
  python -m src.controller.train_ppo --all --profile quick

Training episodes are sampled from the first part of the VALIDATION period (TFT forecasts there are
out-of-sample); the checkpoint is selected on the last part of the validation period - never on test.
Saves: best_model.zip (selected checkpoint), final_model.zip, train_config.json and monitor CSV episode
reward logs (results/logs/ppo/).
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd

from ..common import environment_info, load_all_configs, resolve, save_json, set_global_seed
from ..environment.gym_env import CloudScalingEnv, EnvConfig, feature_names
from ..evaluation.run_experiment import (Context, evaluate_policy, get_traces, load_context, ppo_model_path,
                                         split_traces)
from .policy import PPOPolicy


def make_ppo(env, algo: dict, seed: int):
    from stable_baselines3 import PPO
    lr0, lr1 = float(algo["learning_rate"]), float(algo.get("learning_rate_final", algo["learning_rate"]))
    lr = lambda progress_remaining: lr1 + (lr0 - lr1) * progress_remaining      # linear decay
    return PPO("MlpPolicy", env, learning_rate=lr, target_kl=algo.get("target_kl"), n_steps=int(algo["n_steps"]),
               batch_size=int(algo["batch_size"]), n_epochs=int(algo["n_epochs"]), gamma=float(algo["gamma"]),
               gae_lambda=float(algo["gae_lambda"]), clip_range=float(algo["clip_range"]), ent_coef=float(algo["ent_coef"]),
               vf_coef=float(algo["vf_coef"]), max_grad_norm=float(algo["max_grad_norm"]),
               policy_kwargs={"net_arch": list(algo["net_arch"])}, seed=seed, device="cpu", verbose=0)


def train_variant(ctx: Context, variant: str, seed: int, total_timesteps: int, force: bool = False) -> dict:
    from stable_baselines3.common.callbacks import BaseCallback
    from stable_baselines3.common.monitor import Monitor

    cfgs = ctx.cfgs
    out_dir = ppo_model_path(cfgs, variant, seed).parent
    if (out_dir / "best_model.zip").exists() and not force:
        return {"variant": variant, "seed": seed, "skipped": True}
    out_dir.mkdir(parents=True, exist_ok=True)
    set_global_seed(seed)

    tcfg = cfgs["ppo"]["training"]
    val_traces = get_traces(ctx, tcfg["train_split"] if tcfg["train_split"] != "validation" else "validation")
    train_traces, select_traces = split_traces(val_traces, float(tcfg["train_fraction"]))
    env_cfg = EnvConfig.from_config(cfgs["simulation"], cfgs["ppo"]["variants"][variant])

    log_dir = resolve("results/logs/ppo")
    log_dir.mkdir(parents=True, exist_ok=True)
    env = Monitor(CloudScalingEnv(train_traces, env_cfg, mode="train", seed=seed),
                  filename=str(log_dir / f"{variant}_seed{seed}"))
    model = make_ppo(env, cfgs["ppo"]["algorithm"], seed)
    eval_every = int(tcfg["eval_every_steps"])
    history = []

    class SelectCallback(BaseCallback):
        def __init__(self):
            super().__init__()
            self.best, self.last_eval = -np.inf, 0

        def _select(self):
            tmp = out_dir / "_tmp.zip"
            self.model.save(str(tmp))
            pol = PPOPolicy(tmp, sample_seed=0)
            res = evaluate_policy(pol, select_traces, env_cfg, seed=0)
            score = res["aggregate"]["mean_reward"]
            history.append({"timesteps": self.num_timesteps, "selection_mean_reward": score,
                            "selection_slo_violation": res["aggregate"]["slo_violation_rate_intervals"],
                            "selection_cost": res["aggregate"]["total_cost"]})
            if score > self.best:
                self.best = score
                tmp.replace(out_dir / "best_model.zip")
                self.best_step = self.num_timesteps
            elif tmp.exists():
                tmp.unlink()

        def _on_step(self) -> bool:
            return True

        def _on_rollout_end(self) -> None:
            if self.num_timesteps - self.last_eval >= eval_every:
                self.last_eval = self.num_timesteps
                self._select()

    cb = SelectCallback()
    t0 = time.time()
    model.learn(total_timesteps=int(total_timesteps), callback=cb)
    cb._select()                              # final candidate
    model.save(str(out_dir / "final_model.zip"))
    pd.DataFrame(history).to_csv(out_dir / "selection_history.csv", index=False)
    info = {
        "variant": variant, "seed": seed, "total_timesteps": int(total_timesteps), "train_seconds": time.time() - t0,
        "best_selection_reward": float(cb.best), "best_checkpoint_timesteps": int(getattr(cb, "best_step", total_timesteps)),
        "features": feature_names(env_cfg), "variant_flags": cfgs["ppo"]["variants"][variant],
        "algorithm": cfgs["ppo"]["algorithm"], "reward_weights": cfgs["simulation"]["reward"],
        "actions": cfgs["simulation"]["actions"], "safety": cfgs["simulation"]["safety"],
        "provisioning_delay_steps": env_cfg.sim.delay_steps, "environment": environment_info(),
        "train_traces": [t.name for t in train_traces], "selection_traces": [t.name for t in select_traces],
    }
    save_json(info, out_dir / "train_config.json")
    return info


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", default="ppo_full")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--timesteps", type=int, default=None)
    ap.add_argument("--all", action="store_true", help="train every variant for every seed of --profile")
    ap.add_argument("--profile", default="quick")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--config-dir", default="configs")
    a = ap.parse_args()
    cfgs = load_all_configs(a.config_dir)
    ctx = load_context(cfgs)
    prof = cfgs["experiments"]["profiles"][a.profile]
    steps = a.timesteps or (prof["total_timesteps"] if a.all else cfgs["ppo"]["training"]["total_timesteps"])
    jobs = ([(v, s) for v in cfgs["ppo"]["variants"] for s in prof["seeds"]] if a.all else [(a.variant, a.seed)])
    for v, s in jobs:
        info = train_variant(ctx, v, s, steps, force=a.force)
        if info.get("skipped"):
            print(f"[ppo] {v} seed{s}: exists, skipped")
        else:
            print(f"[ppo] {v} seed{s}: best_sel_reward={info['best_selection_reward']:.4f} "
                  f"at {info['best_checkpoint_timesteps']} steps ({info['train_seconds']:.0f}s)")


if __name__ == "__main__":
    main()
