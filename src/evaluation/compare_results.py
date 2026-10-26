"""Comparison tables, figures and the auto-generated final report (Part G/H).

  python -m src.evaluation.compare_results

Reads results/metrics/runs/*.json (+ forecast/detector evaluation) and writes CSV/Markdown tables to
results/metrics/ and PNG figures to results/figures/. Numbers in docs/final_report.md are inserted
programmatically from these tables - nothing is typed in by hand.
"""
from __future__ import annotations

import glob
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ..common import load_all_configs, load_json, resolve
from ..environment.cost_model import cost_reduction_pct

METRICS = ["total_cost", "mean_replicas_billable", "p95_latency_ms", "p99_latency_ms",
           "slo_violation_rate_intervals", "slo_violation_rate_requests", "unserved_fraction",
           "scaling_events_per_day", "underprovisioning_fraction", "mean_reward"]
LABELS = {"total_cost": "Cost (sim USD / trace)", "p95_latency_ms": "P95 lat (ms)", "p99_latency_ms": "P99 lat (ms)",
          "slo_violation_rate_intervals": "SLO viol. rate", "scaling_events_per_day": "Scaling events/day",
          "underprovisioning_fraction": "Underprov. frac", "mean_reward": "Mean reward", "cost_reduction_pct": "Cost red. vs reactive (%)",
          "unserved_fraction": "Unserved frac", "mean_replicas_billable": "Mean billable replicas"}
FIG = lambda name: resolve("results/figures") / name


# ----------------------------------------------------------------------------------------------
def load_runs() -> pd.DataFrame:
    rows = []
    for f in sorted(glob.glob(str(resolve("results/metrics/runs/*.json")))):
        r = json.load(open(f))
        row = {k: r[k] for k in ("run_id", "policy", "policy_type", "seed", "split", "delay_steps")}
        row.update(r["aggregate"])
        rows.append(row)
    return pd.DataFrame(rows)


def with_cost_reduction(runs: pd.DataFrame, baseline: str = "reactive") -> pd.DataFrame:
    """Add cost reduction (%) vs the named baseline at the same delay (baselines are seed-independent)."""
    runs = runs.copy()
    base = runs[runs["policy"] == baseline].set_index(["split", "delay_steps"])["total_cost"]
    runs["cost_reduction_pct"] = [
        cost_reduction_pct(base.get((s, d), np.nan), c) for s, d, c in zip(runs["split"], runs["delay_steps"], runs["total_cost"])]
    return runs


def summarise(runs: pd.DataFrame, policies: list[str], delay: int, split: str = "test") -> pd.DataFrame:
    sub = runs[(runs["policy"].isin(policies)) & (runs["delay_steps"] == delay) & (runs["split"] == split)]
    cols = METRICS + ["cost_reduction_pct"]
    g = sub.groupby("policy")[cols]
    mean, std, n = g.mean(), g.std(ddof=1).fillna(0.0), g.size()
    out = pd.DataFrame(index=[p for p in policies if p in mean.index])
    for c in cols:
        out[c] = mean.loc[out.index, c]
        out[c + "_std"] = std.loc[out.index, c]
    out["n_seeds"] = n.loc[out.index]
    return out


def to_markdown(df: pd.DataFrame, cols: list[str], fmt: dict | None = None) -> str:
    fmt = fmt or {}
    head = "| policy | " + " | ".join(LABELS.get(c, c) for c in cols) + " | n |"
    sep = "|" + "---|" * (len(cols) + 2)
    lines = [head, sep]
    for p, r in df.iterrows():
        cells = []
        for c in cols:
            f = fmt.get(c, "{:.3f}")
            s = f.format(r[c])
            if r.get("n_seeds", 1) > 1 and r[c + "_std"] > 0:
                s += " ± " + f.format(r[c + "_std"])
            cells.append(s)
        lines.append(f"| {p} | " + " | ".join(cells) + f" | {int(r['n_seeds'])} |")
    return "\n".join(lines)


FMT = {"total_cost": "{:.2f}", "p95_latency_ms": "{:.0f}", "p99_latency_ms": "{:.0f}", "slo_violation_rate_intervals": "{:.2%}",
       "scaling_events_per_day": "{:.1f}", "underprovisioning_fraction": "{:.2%}", "mean_reward": "{:.4f}",
       "cost_reduction_pct": "{:+.1f}", "unserved_fraction": "{:.2%}", "mean_replicas_billable": "{:.1f}"}
MAIN_COLS = ["total_cost", "cost_reduction_pct", "p95_latency_ms", "p99_latency_ms", "slo_violation_rate_intervals",
             "unserved_fraction", "scaling_events_per_day", "underprovisioning_fraction", "mean_reward"]


def save_table(df: pd.DataFrame, name: str, cols: list[str]) -> str:
    df.to_csv(resolve("results/metrics") / f"{name}.csv")
    md = to_markdown(df, cols, FMT)
    (resolve("results/metrics") / f"{name}.md").write_text(md + "\n")
    return md


# ----------------------------------------------------------------------------------------------
# Figures
# ----------------------------------------------------------------------------------------------
def fig_forecast(cfgs: dict) -> None:
    p = resolve("results/metrics/tft_test_predictions.csv")
    if not p.exists():
        return
    d = pd.read_csv(p, parse_dates=["timestamp"])
    g = d[(d["group_id"] == d["group_id"].iloc[0]) & (d["horizon_step"] == 1)].sort_values("timestamp").iloc[:576]
    g12 = d[(d["group_id"] == d["group_id"].iloc[0]) & (d["horizon_step"] == 12)].sort_values("timestamp").iloc[:576]
    fig, ax = plt.subplots(2, 1, figsize=(11, 6), sharex=True)
    for a, gg, title in zip(ax, (g, g12), ("1-step (5 min) ahead", "12-step (60 min) ahead")):
        a.plot(gg["timestamp"], gg["actual"], color="k", lw=1, label="actual demand")
        a.plot(gg["timestamp"], gg["median"], color="tab:blue", lw=1, label="TFT median")
        a.fill_between(gg["timestamp"], gg["lower"], gg["upper"], color="tab:blue", alpha=0.2, label="TFT 10-90% interval")
        a.set_title(f"{title} (test period, first 2 days, {gg['group_id'].iloc[0]})", fontsize=10)
        a.set_ylabel("requests/s")
    ax[0].legend(ncol=3, fontsize=8)
    fig.tight_layout(); fig.savefig(FIG("forecast_examples.png"), dpi=130); plt.close(fig)

    r = load_json("results/metrics/forecast_eval.json")
    fig, ax = plt.subplots(1, 2, figsize=(11, 3.6))
    for name, v in r["methods"].items():
        ax[0].plot(range(1, r["horizon"] + 1), v["mae_by_horizon"], marker="o", ms=3, label=name)
    ax[0].set_xlabel("horizon step (5 min)"); ax[0].set_ylabel("MAE (req/s)"); ax[0].legend(fontsize=8); ax[0].set_title("Forecast error by horizon")
    cov = r["methods"]["tft_median"]["coverage_by_horizon"]
    ax[1].plot(range(1, r["horizon"] + 1), cov, marker="o", ms=3)
    ax[1].axhline(r["methods"]["tft_median"]["interval_nominal_coverage"], color="r", ls="--", label="nominal")
    ax[1].set_ylim(0, 1); ax[1].set_xlabel("horizon step"); ax[1].set_ylabel("coverage"); ax[1].legend(); ax[1].set_title("TFT 10-90% interval coverage")
    fig.tight_layout(); fig.savefig(FIG("forecast_metrics.png"), dpi=130); plt.close(fig)


def fig_tradeoff(summary: pd.DataFrame, name: str, title: str) -> None:
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    for p, r in summary.iterrows():
        ax.errorbar(r["total_cost"], r["slo_violation_rate_intervals"] * 100, xerr=r["total_cost_std"],
                    yerr=r["slo_violation_rate_intervals_std"] * 100, fmt="o", capsize=3, label=p)
    ax.set_xlabel("total simulated cost per trace (USD)"); ax.set_ylabel("SLO-violating intervals (%)")
    ax.set_title(title, fontsize=10); ax.legend(fontsize=8); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(FIG(name), dpi=130); plt.close(fig)


def fig_bars(summary: pd.DataFrame, name: str, title: str, metrics=("total_cost", "slo_violation_rate_intervals", "scaling_events_per_day", "p95_latency_ms")) -> None:
    fig, ax = plt.subplots(1, len(metrics), figsize=(3.2 * len(metrics), 3.6))
    for a, m in zip(np.atleast_1d(ax), metrics):
        a.bar(range(len(summary)), summary[m], yerr=summary[m + "_std"], capsize=3, color="tab:blue", alpha=0.8)
        a.set_xticks(range(len(summary))); a.set_xticklabels(summary.index, rotation=45, ha="right", fontsize=8)
        a.set_title(LABELS.get(m, m), fontsize=9)
    fig.suptitle(title, fontsize=10); fig.tight_layout(); fig.savefig(FIG(name), dpi=130); plt.close(fig)


def fig_delay(runs: pd.DataFrame, policies: list[str]) -> None:
    sub = runs[(runs["policy"].isin(policies)) & (runs["split"] == "test")]
    if sub["delay_steps"].nunique() < 2:
        return
    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    for p in policies:
        s = sub[sub["policy"] == p].groupby("delay_steps")
        if s.ngroups == 0:
            continue
        m, sd = s["slo_violation_rate_intervals"].mean() * 100, s["slo_violation_rate_intervals"].std(ddof=1).fillna(0) * 100
        ax[0].errorbar(m.index, m.values, yerr=sd.values, marker="o", capsize=3, label=p)
        m2, sd2 = s["total_cost"].mean(), s["total_cost"].std(ddof=1).fillna(0)
        ax[1].errorbar(m2.index, m2.values, yerr=sd2.values, marker="o", capsize=3, label=p)
    ax[0].set_xlabel("provisioning delay at test time (steps of 5 min)"); ax[0].set_ylabel("SLO-violating intervals (%)")
    ax[1].set_xlabel("provisioning delay at test time (steps of 5 min)"); ax[1].set_ylabel("cost (sim USD / trace)")
    ax[0].legend(fontsize=7); ax[0].set_title("Delay sensitivity (trained with delay=3)", fontsize=10); ax[1].set_title("Cost vs delay", fontsize=10)
    for a in ax: a.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(FIG("delay_sensitivity.png"), dpi=130); plt.close(fig)


def fig_timeline(policies: list[tuple[str, str]], delay: int, n_steps: int = 576) -> None:
    fig, ax = plt.subplots(len(policies), 1, figsize=(11, 2.6 * len(policies)), sharex=True)
    for a, (label, rid) in zip(np.atleast_1d(ax), policies):
        p = resolve("results/logs") / f"steps_{rid}.parquet"
        if not p.exists():
            continue
        s = pd.read_parquet(p)
        s = s[s["trace"] == s["trace"].iloc[0]].iloc[:n_steps]
        a.plot(s["timestamp"], s["demand"] / 100.0, color="k", lw=1, label="demand / capacity-per-replica")
        a.step(s["timestamp"], s["ready"], where="post", color="tab:green", label="ready replicas")
        a.step(s["timestamp"], s["ready"] + s["pending"], where="post", color="tab:orange", ls="--", lw=1, label="ready+pending")
        viol = s[s["violated"]]
        a.scatter(viol["timestamp"], np.full(len(viol), 0.5), marker="|", color="r", s=30, label="SLO violation")
        a.set_title(label, fontsize=9); a.set_ylabel("replicas")
    np.atleast_1d(ax)[0].legend(ncol=4, fontsize=7)
    fig.tight_layout(); fig.savefig(FIG("scaling_timelines.png"), dpi=130); plt.close(fig)


def fig_learning_curves(variants: list[str]) -> None:
    fig, ax = plt.subplots(1, 2, figsize=(11, 3.8))
    for v in variants:
        for f in sorted(glob.glob(str(resolve(f"results/logs/ppo/{v}_seed*.monitor.csv")))):
            d = pd.read_csv(f, skiprows=1)
            if len(d) == 0:
                continue
            r = d["r"].rolling(10, min_periods=1).mean()
            ax[0].plot(d["l"].cumsum(), r, alpha=0.7, label=v if f.endswith("seed0.monitor.csv") else None)
    ax[0].set_xlabel("environment steps"); ax[0].set_ylabel("episode reward (10-ep mean)"); ax[0].set_title("PPO training episode reward", fontsize=10); ax[0].legend(fontsize=7)
    for v in variants:
        for f in sorted(glob.glob(str(resolve(f"models/controller/{v}/seed*/selection_history.csv")))):
            d = pd.read_csv(f)
            ax[1].plot(d["timesteps"], d["selection_mean_reward"], alpha=0.7, label=v if "seed0" in f else None)
    ax[1].set_xlabel("environment steps"); ax[1].set_ylabel("mean reward on selection traces"); ax[1].set_title("Checkpoint-selection curve", fontsize=10); ax[1].legend(fontsize=7)
    fig.tight_layout(); fig.savefig(FIG("ppo_learning_curves.png"), dpi=130); plt.close(fig)


def fig_anomaly(cfgs: dict) -> None:
    sp = resolve(cfgs["anomaly"]["paths"]["scores_file"])
    ev = resolve("results/metrics/detector_eval.json")
    if not sp.exists() or not ev.exists():
        return
    sc = pd.read_parquet(sp)
    t = sc[(sc["split"] == "test") & (sc["group_id"] == sc["group_id"].iloc[0])].sort_values("time_idx").iloc[:576]
    fig, ax = plt.subplots(2, 1, figsize=(11, 5), sharex=True)
    ax[0].plot(t["timestamp"], t["z_raw"], lw=0.8, label="raw-metric score"); ax[0].plot(t["timestamp"], t["z_resid"], lw=0.8, label="residual score")
    for _, r in t[t["anomaly_label"] == 1].iterrows():
        ax[0].axvspan(r["timestamp"], r["timestamp"] + pd.Timedelta(minutes=5), color="r", alpha=0.15, lw=0)
    ax[0].set_ylabel("normalised IF score"); ax[0].legend(fontsize=8); ax[0].set_title("Isolation Forest scores (test; red = synthetic anomaly labels)", fontsize=10)
    r = load_json("results/metrics/detector_eval.json")
    names = ["raw_metrics", "forecast_residuals", "union_of_both"]
    w = 0.25
    for i, m in enumerate(["precision", "recall", "f1"]):
        ax[1].bar(np.arange(3) + i * w, [r[n][m] for n in names], w, label=m)
    ax[1].set_xticks(np.arange(3) + w); ax[1].set_xticklabels(names); ax[1].legend(fontsize=8); ax[1].set_ylim(0, 1)
    fig.tight_layout(); fig.savefig(FIG("anomaly_detection.png"), dpi=130); plt.close(fig)


def fig_architecture() -> None:
    fig, ax = plt.subplots(figsize=(12, 5.6)); ax.axis("off"); ax.set_xlim(0, 12); ax.set_ylim(0, 5.6)
    def box(x, y, w, h, text, color):
        ax.add_patch(plt.Rectangle((x, y), w, h, fc=color, ec="k", lw=1, alpha=0.85))
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=8)
    def arrow(x1, y1, x2, y2):
        ax.annotate("", xy=(x2, y2), xytext=(x1, y1), arrowprops=dict(arrowstyle="->", lw=1.2))
    box(0.2, 2.3, 1.9, 1.0, "Workload data\nCPU, memory, demand,\nlatency, replicas, cost", "#dde8f7")
    box(2.6, 2.3, 1.9, 1.0, "Data pipeline\nvalidate · clean ·\nchronological split", "#dde8f7")
    box(5.0, 3.8, 2.1, 1.1, "TFT forecaster\nquantiles q10/q50/q90\n→ demand + uncertainty", "#e6f2dc")
    box(5.0, 0.7, 2.1, 1.1, "Isolation Forest\nraw metrics / TFT\nresiduals → anomaly z", "#e6f2dc")
    box(7.6, 2.2, 2.0, 1.2, "PPO controller\nbounded actions\n(-1, 0, +1, +3)", "#fbe6c8")
    box(7.6, 0.2, 2.0, 0.9, "Safety shield\nbounds · feasibility", "#f7d6d6")
    box(10.0, 2.2, 1.8, 1.2, "Resource simulator\ncapacity · delay ·\ncost · latency · SLO", "#e8dff5")
    box(10.0, 4.0, 1.8, 1.0, "Evaluation\nbaselines · ablations\ndashboard", "#eeeeee")
    arrow(2.1, 2.8, 2.6, 2.8); arrow(4.5, 3.0, 5.0, 4.2); arrow(4.5, 2.6, 5.0, 1.4)
    arrow(7.1, 4.2, 7.6, 3.2); arrow(7.1, 1.4, 7.6, 2.4); arrow(8.6, 2.2, 8.6, 1.1)
    arrow(9.6, 0.65, 10.4, 2.2); arrow(9.6, 2.8, 10.0, 2.8); arrow(10.9, 3.4, 10.9, 4.0)
    arrow(10.0, 2.5, 9.6, 2.5)
    ax.text(10.9, 1.8, "state: demand, pending\ncapacity, cost", ha="center", fontsize=7)
    ax.set_title("AI Cloud Cost Prediction and Intelligent Resource Allocation - architecture", fontsize=11)
    fig.tight_layout(); fig.savefig(resolve("docs/architecture.png"), dpi=140); plt.close(fig)
    fig2 = None


# ----------------------------------------------------------------------------------------------
def build_all() -> dict:
    cfgs = load_all_configs()
    resolve("results/figures").mkdir(parents=True, exist_ok=True)
    runs = load_runs()
    out = {}
    fig_architecture()
    fig_forecast(cfgs)
    if runs.empty:
        return out
    runs = with_cost_reduction(runs, "reactive")
    runs.to_csv(resolve("results/metrics/all_runs.csv"), index=False)
    d0 = int(cfgs["simulation"]["provisioning"]["delay_steps"])
    ppo_all = sorted(v for v in cfgs["ppo"]["variants"])
    base = ["fixed", "fixed_peak", "reactive", "predictive", "predictive_q"]

    out["exp2"] = save_table(summarise(runs, base, d0), "table_exp2_baselines", MAIN_COLS)
    fig_tradeoff(summarise(runs, base + ["ppo_full"], d0), "tradeoff_all.png", f"Cost vs reliability (test, delay={d0})")
    fig_bars(summarise(runs, base + ["ppo_full"], d0), "bars_main.png", "Policy comparison (test traces)")
    out["exp3"] = save_table(summarise(runs, ["reactive", "predictive", "predictive_q", "ppo_point", "ppo_full", "ppo_base", "ppo_unc"], d0),
                             "table_exp3_uncertainty", MAIN_COLS)
    out["exp4"] = save_table(summarise(runs, ["ppo_base", "ppo_point", "ppo_unc", "ppo_full"], d0), "table_exp4_pending", MAIN_COLS)
    out["exp5"] = save_table(summarise(runs, ["ppo_full", "ppo_full_rawz", "ppo_full_residz"], d0), "table_exp5_anomaly", MAIN_COLS)
    delay_pols = ["reactive", "predictive", "ppo_base", "ppo_point", "ppo_unc", "ppo_full"]
    fig_delay(runs, delay_pols)
    rows = []
    for d in sorted(runs["delay_steps"].unique()):
        s = summarise(runs, delay_pols, int(d))
        for p, r in s.iterrows():
            rows.append({"delay_steps": d, "policy": p, "total_cost": r["total_cost"], "total_cost_std": r["total_cost_std"],
                         "slo_violation_rate_intervals": r["slo_violation_rate_intervals"],
                         "slo_violation_rate_intervals_std": r["slo_violation_rate_intervals_std"],
                         "scaling_events_per_day": r["scaling_events_per_day"], "mean_reward": r["mean_reward"], "n_seeds": r["n_seeds"]})
    pd.DataFrame(rows).to_csv(resolve("results/metrics/table_exp4_delay_sensitivity.csv"), index=False)
    fig_bars(summarise(runs, ["ppo_point", "ppo_full", "ppo_base", "ppo_unc"], d0), "bars_exp3_exp4.png", "Uncertainty / pending-state ablation")
    fig_bars(summarise(runs, ["ppo_full", "ppo_full_rawz", "ppo_full_residz"], d0), "bars_exp5.png", "Anomaly-signal ablation")
    fig_timeline([("Reactive (tuned)", f"reactive_test_d{d0}"), ("Predictive (TFT median)", f"predictive_test_d{d0}"),
                  ("PPO uncertainty+pending (seed 0)", f"ppo_full_s0_test_d{d0}")], d0)
    fig_learning_curves(["ppo_base", "ppo_point", "ppo_unc", "ppo_full"])
    fig_anomaly(cfgs)
    return out


def main() -> None:
    out = build_all()
    for k, v in out.items():
        print(f"\n== {k} ==\n{v}")
    print("\nfigures ->", resolve("results/figures"))


if __name__ == "__main__":
    main()
