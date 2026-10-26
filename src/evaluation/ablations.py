"""Experiments 1-5 and ablation studies (Part G).

  python -m src.evaluation.ablations --profile quick            # everything (trains missing PPO runs)
  python -m src.evaluation.ablations --profile quick --only exp4

Exp1 forecasting | Exp2 predictive vs reactive | Exp3 uncertainty-aware PPO | Exp4 provisioning delay /
pending-capacity state | Exp5 anomaly signal.  PPO runs use several seeds; all seeds are reported.
"""
from __future__ import annotations

import argparse
import time

from ..common import load_all_configs, save_json
from .run_experiment import Context, load_context, run_and_save, tune_baselines

BASELINES = ["fixed", "fixed_peak", "reactive", "predictive", "predictive_q"]
EXP_VARIANTS = {
    "exp3": ["ppo_point", "ppo_full"],                         # point forecast vs quantile forecast (pending on)
    "exp3b": ["ppo_base", "ppo_unc"],                          # same comparison without pending state
    "exp4": ["ppo_base", "ppo_point", "ppo_unc", "ppo_full"],  # 2x2: uncertainty x pending
    "exp5": ["ppo_full", "ppo_full_rawz", "ppo_full_residz"],
}


def _default_delay(ctx: Context) -> int:
    return int(ctx.cfgs["simulation"]["provisioning"]["delay_steps"])


def ensure_trained(ctx: Context, variants: list[str], seeds: list[int], timesteps: int, force: bool = False):
    from ..controller.train_ppo import train_variant
    for v in variants:
        for s in seeds:
            info = train_variant(ctx, v, s, timesteps, force=force)
            if not info.get("skipped"):
                print(f"  [train] {v} seed{s}: sel_reward={info['best_selection_reward']:.4f} ({info['train_seconds']:.0f}s)", flush=True)


def exp1_forecasting(ctx: Context):
    from ..forecasting.evaluate import evaluate_forecasts
    return evaluate_forecasts(ctx.cfgs)


def exp2_baselines(ctx: Context):
    tune_baselines(ctx)
    return [run_and_save(ctx, b, split="test", delay=_default_delay(ctx)) for b in BASELINES]


def run_ppo_set(ctx: Context, variants: list[str], seeds: list[int], delays: list[int]):
    out = []
    for v in variants:
        for s in seeds:
            for d in delays:
                out.append(run_and_save(ctx, "ppo", variant=v, seed=s, split="test", delay=d))
    return out


def exp5_detector(ctx: Context):
    from ..anomaly.evaluate_detector import evaluate_detectors
    return evaluate_detectors(ctx.cfgs)


def run_ablations(profile: str = "quick", only: str | None = None, force_train: bool = False, cfgs: dict | None = None):
    cfgs = cfgs or load_all_configs()
    ctx = load_context(cfgs)
    prof = cfgs["experiments"]["profiles"][profile]
    seeds, steps = list(prof["seeds"]), int(prof["total_timesteps"])
    d0 = _default_delay(ctx)
    delays = [d for d in cfgs["experiments"]["evaluation"]["provisioning_delays"]]
    t0 = time.time()
    want = lambda k: only in (None, k)

    if want("exp1"):
        print("[exp1] forecasting", flush=True); exp1_forecasting(ctx)
    if want("exp2"):
        print("[exp2] baselines (tuned on validation)", flush=True); exp2_baselines(ctx)
    if only in (None, "exp3", "exp4", "exp5"):
        variants = sorted({v for k in ("exp3", "exp3b", "exp4", "exp5") for v in EXP_VARIANTS[k]})
        print(f"[train] PPO variants {variants} x seeds {seeds} x {steps} steps", flush=True)
        ensure_trained(ctx, variants, seeds, steps, force_train)
        print("[eval] PPO at default delay", flush=True)
        run_ppo_set(ctx, variants, seeds, [d0])
    if want("exp4"):
        print("[exp4] delay sensitivity", flush=True)
        for d in delays:
            if d == d0:
                continue
            for b in BASELINES:
                run_and_save(ctx, b, split="test", delay=d)
        run_ppo_set(ctx, EXP_VARIANTS["exp4"], seeds, [d for d in delays if d != d0])
    if want("exp5"):
        print("[exp5] detector evaluation", flush=True); exp5_detector(ctx)
    save_json({"profile": profile, "seeds": seeds, "ppo_total_timesteps": steps, "delays_evaluated": delays,
               "default_delay": d0, "wall_seconds": time.time() - t0}, "results/metrics/experiments_manifest.json")
    print(f"[done] {time.time() - t0:.0f}s", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--profile", default="quick")
    ap.add_argument("--only", default=None, choices=[None, "exp1", "exp2", "exp3", "exp4", "exp5"])
    ap.add_argument("--force-train", action="store_true")
    a = ap.parse_args()
    run_ablations(a.profile, a.only, a.force_train)


if __name__ == "__main__":
    main()
