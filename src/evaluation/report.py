"""Generate docs/final_report.md from the measured results (no hand-typed numbers).

Statements are produced by rules: a difference between two PPO variants is called "within seed
variability" unless the gap between means exceeds the larger of the two standard deviations.
"""
from __future__ import annotations

import json

import pandas as pd

from ..common import load_json, resolve


def _t(name: str) -> pd.DataFrame | None:
    p = resolve(f"results/metrics/{name}.csv")
    return pd.read_csv(p, index_col=0) if p.exists() else None


def _md(name: str) -> str:
    p = resolve(f"results/metrics/{name}.md")
    return p.read_text() if p.exists() else "_not available_"


def cmp(df: pd.DataFrame, a: str, b: str, col: str, fmt: str = "{:.2f}", lower_better: bool = True, unit: str = "") -> str:
    if df is None or a not in df.index or b not in df.index:
        return f"- {a} vs {b} ({col}): not available."
    ma, mb = df.loc[a, col], df.loc[b, col]
    sa, sb = df.loc[a, col + "_std"], df.loc[b, col + "_std"]
    gap = ma - mb
    distinct = abs(gap) > max(sa, sb)
    better = (gap < 0) == lower_better
    verdict = ("within seed variability (no reliable difference)" if not distinct else
               f"{a} is {'better' if better else 'worse'}")
    f = lambda x: fmt.format(x) + unit
    return f"- **{col}**: {a} {f(ma)} (±{fmt.format(sa)}) vs {b} {f(mb)} (±{fmt.format(sb)}) -> {verdict}."


def write_report() -> str:
    man = load_json("results/metrics/experiments_manifest.json") if resolve("results/metrics/experiments_manifest.json").exists() else {}
    fe = load_json("results/metrics/forecast_eval.json") if resolve("results/metrics/forecast_eval.json").exists() else None
    de = load_json("results/metrics/detector_eval.json") if resolve("results/metrics/detector_eval.json").exists() else None
    e2, e3, e4, e5 = (_t(n) for n in ("table_exp2_baselines", "table_exp3_uncertainty", "table_exp4_pending", "table_exp5_anomaly"))
    L = []
    L += ["# Final report (auto-generated from measured results)", "",
          "> **Scope.** Synthetic workload, simulated cost/latency, simulator-approximated reactive baseline. Nothing here is an",
          "> invoice or a production measurement, and nothing here shows superiority over a real Kubernetes HPA.", ""]
    if man:
        L += [f"Profile `{man['profile']}`: PPO seeds {man['seeds']}, {man['ppo_total_timesteps']} training steps per run, "
              f"provisioning delays evaluated {man['delays_evaluated']} (trained with {man['default_delay']}).", ""]
    L += ["## Experiment 1 - Forecasting", ""]
    if fe:
        m = fe["methods"]
        L += ["| method | MAE (req/s) | RMSE (req/s) |", "|---|---|---|"]
        L += [f"| {k} | {v['mae']:.2f} | {v['rmse']:.2f} |" for k, v in m.items()]
        t = m["tft_median"]
        L += ["", f"- Test windows: {fe['n_windows']} (12-step horizon, all targets inside the test period).",
              f"- TFT MAE is {fe['tft_vs_best_baseline_mae_improvement_pct']:.1f}% lower than the best baseline "
              f"(`{fe['best_baseline_by_mae']}`) and {fe['tft_vs_persistence_mae_improvement_pct']:.1f}% lower than persistence.",
              f"- Mean quantile loss {t['quantile_loss_mean']:.2f}; 10-90% interval coverage {t['interval_coverage']:.1%} "
              f"(nominal {t['interval_nominal_coverage']:.0%}), mean width {t['interval_mean_width']:.1f} req/s"
              + (" - the intervals are **narrower than nominal** (under-coverage)." if t['interval_coverage'] < t['interval_nominal_coverage'] - 0.03 else "."),
              "", "![forecast](../results/figures/forecast_examples.png)", "![forecast metrics](../results/figures/forecast_metrics.png)", ""]
    L += ["## Experiment 2 - Predictive versus reactive scaling (test traces)", "", _md("table_exp2_baselines"), "",
          "Cost reduction is relative to the tuned reactive baseline; positive = cheaper. Read it together with SLO violations and tail latency.", ""]
    if e2 is not None and {"reactive", "predictive"} <= set(e2.index):
        r, p = e2.loc["reactive"], e2.loc["predictive"]
        L += [f"- Predictive vs reactive: cost {p['cost_reduction_pct']:+.1f}%, SLO-violating intervals {p['slo_violation_rate_intervals']:.2%} vs "
              f"{r['slo_violation_rate_intervals']:.2%}, P95 {p['p95_latency_ms']:.0f} vs {r['p95_latency_ms']:.0f} ms, "
              f"churn {p['scaling_events_per_day']:.1f} vs {r['scaling_events_per_day']:.1f} events/day.", ""]
    L += ["![tradeoff](../results/figures/tradeoff_all.png)", "![bars](../results/figures/bars_main.png)", "![timelines](../results/figures/scaling_timelines.png)", "",
          "## Experiment 3 - Uncertainty-aware PPO", "", _md("table_exp3_uncertainty"), "", "Point forecast vs TFT quantiles (pending state available in both):"]
    for c, f, lb in (("total_cost", "{:.2f}", True), ("slo_violation_rate_intervals", "{:.2%}", True), ("mean_reward", "{:.4f}", False), ("scaling_events_per_day", "{:.1f}", True)):
        L.append(cmp(e3, "ppo_full", "ppo_point", c, f, lb))
    L += ["", "Same comparison without pending state:"]
    for c, f, lb in (("total_cost", "{:.2f}", True), ("slo_violation_rate_intervals", "{:.2%}", True), ("mean_reward", "{:.4f}", False)):
        L.append(cmp(e3, "ppo_unc", "ppo_base", c, f, lb))
    L += ["", "![exp3/4](../results/figures/bars_exp3_exp4.png)", "![learning](../results/figures/ppo_learning_curves.png)", "",
          "## Experiment 4 - Provisioning delay / pending-capacity state", "", _md("table_exp4_pending"), "",
          "Pending state on vs off:"]
    for c, f, lb in (("total_cost", "{:.2f}", True), ("slo_violation_rate_intervals", "{:.2%}", True), ("mean_reward", "{:.4f}", False)):
        L.append(cmp(e4, "ppo_full", "ppo_unc", c, f, lb))
        L.append(cmp(e4, "ppo_point", "ppo_base", c, f, lb))
    p = resolve("results/metrics/table_exp4_delay_sensitivity.csv")
    ds = pd.read_csv(p) if p.exists() else None
    if ds is not None:
        piv = ds.pivot(index="policy", columns="delay_steps", values="slo_violation_rate_intervals") * 100
        L += ["", "SLO-violating intervals (%) when the provisioning delay at test time differs from the training delay (3):", "",
              piv.round(2).to_markdown(), "", "![delay](../results/figures/delay_sensitivity.png)"]
    L += ["", "## Experiment 5 - Anomaly signal", ""]
    if de:
        L += ["Detector quality on the test split (synthetic labels):", "", "| config | precision | recall | F1 | FPR | AUROC |", "|---|---|---|---|---|---|"]
        for k in ("raw_metrics", "forecast_residuals", "union_of_both"):
            v = de[k]
            L.append(f"| {k} | {v['precision']:.3f} | {v['recall']:.3f} | {v['f1']:.3f} | {v['fpr']:.4f} | {v.get('auroc', float('nan')):.3f} |")
        L.append("")
    L += [_md("table_exp5_anomaly"), "", "Controller with raw-metric score / residual score vs no anomaly signal:"]
    for a in ("ppo_full_rawz", "ppo_full_residz"):
        for c, f, lb in (("total_cost", "{:.2f}", True), ("slo_violation_rate_intervals", "{:.2%}", True), ("mean_reward", "{:.4f}", False)):
            L.append(cmp(e5, a, "ppo_full", c, f, lb))
    L += ["", "![anomaly](../results/figures/anomaly_detection.png)", "![bars exp5](../results/figures/bars_exp5.png)", "",
          "## Limitations", "", "See `docs/limitations.md`. In particular: synthetic data, simulated latency/cost, small PPO budget and few seeds.", ""]
    text = "\n".join(L)
    resolve("docs/final_report.md").write_text(text)
    return text


if __name__ == "__main__":
    write_report()
