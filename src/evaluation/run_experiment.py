"""Repeatable policy evaluation (Part G).

CLI:
  python -m src.evaluation.run_experiment --policy reactive
  python -m src.evaluation.run_experiment --policy ppo --variant ppo_full --seed 0
Every run saves its configuration snapshot, seed, library versions and per-trace + aggregate metrics to
results/metrics/runs/<run_id>.json (and the step log to results/logs/) so results can be reproduced.
All policies run on the SAME held-out traces and the same resource constraints.
"""
from __future__ import annotations

import argparse
import itertools
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from ..baselines.fixed_policy import FixedPolicy, size_fixed
from ..baselines.predictive_policy import PredictivePolicy
from ..baselines.reactive_policy import ReactivePolicy
from ..common import config_hash, environment_info, load_all_configs, resolve, save_json
from ..controller.policy import BasePolicy, PPOPolicy
from ..environment.gym_env import CloudScalingEnv, EnvConfig
from ..environment.workload import WorkloadTrace, traces_from_data
from .metrics import aggregate_traces, episode_metrics

POLICIES = ["fixed", "fixed_peak", "reactive", "predictive", "predictive_q", "ppo"]


# ----------------------------------------------------------------------------------------------
@dataclass
class Context:
    cfgs: dict
    df: pd.DataFrame
    meta: dict
    predictions: pd.DataFrame | None
    scores: pd.DataFrame | None
    _cache: dict


def load_context(cfgs: dict | None = None) -> Context:
    cfgs = cfgs or load_all_configs()
    from ..data.load_data import load_processed
    df = load_processed(cfgs["data"])
    meta = json.load(open(resolve("data/processed/preprocess_config.json")))
    pp = resolve(cfgs["forecasting"]["paths"]["predictions_file"])
    sp = resolve(cfgs["anomaly"]["paths"]["scores_file"])
    return Context(cfgs, df, meta, pd.read_parquet(pp) if pp.exists() else None,
                   pd.read_parquet(sp) if sp.exists() else None, {})


def get_traces(ctx: Context, split: str) -> list[WorkloadTrace]:
    if split not in ctx._cache:
        ctx._cache[split] = traces_from_data(ctx.df, ctx.predictions, ctx.scores, split,
                                             horizon=int(ctx.cfgs["forecasting"]["horizon"]))
    return ctx._cache[split]


def split_traces(traces: list[WorkloadTrace], frac: float) -> tuple[list, list]:
    """First `frac` of each trace (e.g. PPO training) and the remainder (e.g. checkpoint selection)."""
    a, b = [], []
    for t in traces:
        k = int(len(t) * frac)
        a.append(t.slice(0, k, f"{t.name}[train]"))
        b.append(t.slice(k, len(t), f"{t.name}[select]"))
    return a, b


# ----------------------------------------------------------------------------------------------
def rollout(policy: BasePolicy, traces: list[WorkloadTrace], env_cfg: EnvConfig, seed: int = 0) -> list[pd.DataFrame]:
    """Run `policy` over each trace from its first to last valid origin; return per-trace step logs."""
    env = CloudScalingEnv(traces, env_cfg, mode="eval", seed=seed)
    logs = []
    for k in range(len(traces)):
        n0 = policy.initial_replicas(None)
        obs, info = env.reset(seed=seed + k, options={"trace_index": k, **({"initial_replicas": n0} if n0 else {})})
        policy.reset(env.decision_state())
        rows, done = [], False
        while not done:
            state = env.decision_state()
            a = policy.act(state, obs)
            ts = env.trace.timestamps[env.i + 1]
            obs, r, term, trunc, inf = env.step(a)
            rows.append({"trace": env.trace.name, "timestamp": ts, "demand": inf["demand"], "ready": inf["ready"],
                         "pending": inf["pending"], "billable": inf["billable"], "cost": inf["cost"],
                         "latency_ms": inf["latency_ms"], "rho": inf["rho"], "dropped": inf["dropped"],
                         "violated": inf["violated"], "under_provisioned": inf["under_provisioned"],
                         "applied_delta": inf["applied_delta"], "safety_intervened": inf["safety_intervened"],
                         "cpu_utilization": inf["cpu_utilization"], "reward": r})
            done = term or trunc
        logs.append(pd.DataFrame(rows))
    return logs


def evaluate_policy(policy: BasePolicy, traces: list[WorkloadTrace], env_cfg: EnvConfig, seed: int = 0) -> dict:
    logs = rollout(policy, traces, env_cfg, seed)
    per = [episode_metrics(l, env_cfg.sim.interval_minutes) for l in logs]
    for p, t in zip(per, traces):
        p["trace"] = t.name
    return {"per_trace": per, "aggregate": aggregate_traces([{k: v for k, v in p.items() if k != "trace"} for p in per]),
            "steps": pd.concat(logs, ignore_index=True)}


# ----------------------------------------------------------------------------------------------
def ppo_model_path(cfgs: dict, variant: str, seed: int) -> Path:
    return resolve(cfgs["ppo"]["paths"]["model_dir"]) / variant / f"seed{seed}" / "best_model.zip"


def train_demand(ctx: Context) -> np.ndarray:
    return ctx.df.loc[ctx.df["split"] == "train", "request_rate"].to_numpy(float)


def default_params(name: str) -> dict:
    return {"reactive": {"target_util": 0.7, "tolerance": 0.1, "scale_down_stabilization_steps": 6},
            "predictive": {"target_util": 0.7, "scale_down_stabilization_steps": 3},
            "predictive_q": {"target_util": 0.7, "scale_down_stabilization_steps": 3}}.get(name, {})


def load_tuned(name: str) -> dict:
    p = resolve("results/metrics/tuned_baselines.json")
    if p.exists():
        d = json.load(open(p))
        if name in d:
            return d[name]["params"]
    return default_params(name)


def build_policy(name: str, ctx: Context, params: dict | None = None, variant: str | None = None,
                 seed: int = 0, sim_cfg: dict | None = None) -> BasePolicy:
    sc = sim_cfg or ctx.cfgs["simulation"]
    ex = ctx.cfgs["experiments"]["baselines"]
    cap = EnvConfig.from_config(sc).sim.capacity_per_replica
    lo, hi = sc["cluster"]["min_replicas"], sc["cluster"]["max_replicas"]
    if name == "fixed":
        n = size_fixed(train_demand(ctx), cap, ex["fixed"]["target_util"], ex["fixed"]["sizing_quantile"], lo, hi)
        return FixedPolicy(n, "fixed")
    if name == "fixed_peak":
        n = size_fixed(train_demand(ctx), cap, ex["fixed_peak"]["target_util"], None, lo, hi)
        return FixedPolicy(n, "fixed_peak")
    if name == "reactive":
        return ReactivePolicy(**(params or load_tuned("reactive")))
    if name in ("predictive", "predictive_q"):
        return PredictivePolicy(use_quantile="median" if name == "predictive" else "upper",
                                **(params or load_tuned(name)))
    if name == "ppo":
        variant = variant or "ppo_full"
        return PPOPolicy(ppo_model_path(ctx.cfgs, variant, seed), name=variant, sample_seed=seed)
    raise ValueError(f"unknown policy {name}")


def env_cfg_for(ctx: Context, policy_name: str, variant: str | None, delay: int | None = None) -> EnvConfig:
    v = ctx.cfgs["ppo"]["variants"][variant] if policy_name == "ppo" else {}
    return EnvConfig.from_config(ctx.cfgs["simulation"], v, delay_steps=delay)


# ----------------------------------------------------------------------------------------------
def tune_baselines(ctx: Context) -> dict:
    """Grid-search baseline hyper-parameters on the VALIDATION traces (never test), maximising mean reward."""
    traces = get_traces(ctx, ctx.cfgs["experiments"]["tuning_split"])
    env_cfg = EnvConfig.from_config(ctx.cfgs["simulation"])
    out = {}
    for name in ("reactive", "predictive", "predictive_q"):
        grid = ctx.cfgs["experiments"]["baselines"][name]["grid"]
        keys = list(grid)
        best, table = None, []
        for vals in itertools.product(*[grid[k] for k in keys]):
            params = dict(zip(keys, vals))
            pol = build_policy(name, ctx, params=params)
            res = evaluate_policy(pol, traces, env_cfg, seed=0)
            a = res["aggregate"]
            table.append({**params, "mean_reward": a["mean_reward"], "total_cost": a["total_cost"],
                          "slo_violation_rate_intervals": a["slo_violation_rate_intervals"]})
            if best is None or a["mean_reward"] > best[0]:
                best = (a["mean_reward"], params)
        out[name] = {"params": best[1], "validation_mean_reward": best[0], "grid_results": table}
    save_json({**out, "tuned_on": ctx.cfgs["experiments"]["tuning_split"]}, "results/metrics/tuned_baselines.json")
    return out


def run_and_save(ctx: Context, policy_name: str, variant: str | None = None, seed: int = 0, split: str = "test",
                 delay: int | None = None, tag: str = "", metrics_dir: str = "results/metrics/runs",
                 logs_dir: str = "results/logs") -> dict:
    cfgs = ctx.cfgs
    env_cfg = env_cfg_for(ctx, policy_name, variant, delay)
    policy = build_policy(policy_name, ctx, variant=variant, seed=seed)
    traces = get_traces(ctx, split)
    res = evaluate_policy(policy, traces, env_cfg, seed=seed)
    name = variant if policy_name == "ppo" else policy_name
    d = env_cfg.sim.delay_steps
    run_id = f"{name}{'_s' + str(seed) if policy_name == 'ppo' else ''}_{split}_d{d}{tag}"
    record = {
        "run_id": run_id, "policy": name, "policy_type": policy_name, "variant": variant, "seed": seed,
        "split": split, "delay_steps": d,
        "params": {k: v for k, v in vars(policy).items() if k in ("target", "tol", "K", "q", "window", "replicas")},
        "aggregate": res["aggregate"], "per_trace": res["per_trace"],
        "config_hashes": {k: config_hash(v) for k, v in cfgs.items()},
        "config_snapshot": {"simulation": cfgs["simulation"], "ppo_variant": (cfgs["ppo"]["variants"].get(variant) if variant else None)},
        "environment": environment_info(), "n_traces": len(traces),
    }
    save_json(record, Path(metrics_dir) / f"{run_id}.json")
    steps_path = resolve(logs_dir) / f"steps_{run_id}.parquet"
    steps_path.parent.mkdir(parents=True, exist_ok=True)
    res["steps"].to_parquet(steps_path, index=False)
    return record


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--policy", required=True, choices=POLICIES)
    ap.add_argument("--variant", default="ppo_full")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--split", default="test")
    ap.add_argument("--delay", type=int, default=None, help="override provisioning delay (steps)")
    ap.add_argument("--config-dir", default="configs")
    a = ap.parse_args()
    ctx = load_context(load_all_configs(a.config_dir))
    rec = run_and_save(ctx, a.policy, a.variant if a.policy == "ppo" else None, a.seed, a.split, a.delay)
    g = rec["aggregate"]
    print(f"[{rec['run_id']}] cost={g['total_cost']:.3f} USD(sim)  p95={g['p95_latency_ms']:.0f}ms  "
          f"slo_viol={g['slo_violation_rate_intervals']:.3%}  churn/day={g['scaling_events_per_day']:.1f}  "
          f"underprov={g['underprovisioning_fraction']:.3%}")


if __name__ == "__main__":
    main()
