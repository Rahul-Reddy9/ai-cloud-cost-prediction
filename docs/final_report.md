# Final report (auto-generated from measured results)

> **Scope.** Synthetic workload, simulated cost/latency, simulator-approximated reactive baseline. Nothing here is an
> invoice or a production measurement, and nothing here shows superiority over a real Kubernetes HPA.

Profile `quick`: PPO seeds [0, 1, 2], 150000 training steps per run, provisioning delays evaluated [0, 1, 3, 6] (trained with 3).

## Experiment 1 - Forecasting

| method | MAE (req/s) | RMSE (req/s) |
|---|---|---|
| moving_average | 43.43 | 80.93 |
| persistence | 31.27 | 83.39 |
| seasonal_naive | 41.72 | 97.36 |
| tft_median | 21.77 | 64.87 |

- Test windows: 8585 (12-step horizon, all targets inside the test period).
- TFT MAE is 30.4% lower than the best baseline (`persistence`) and 30.4% lower than persistence.
- Mean quantile loss 8.10; 10-90% interval coverage 73.6% (nominal 80%), mean width 35.2 req/s - the intervals are **narrower than nominal** (under-coverage).

![forecast](../results/figures/forecast_examples.png)
![forecast metrics](../results/figures/forecast_metrics.png)

## Experiment 2 - Predictive versus reactive scaling (test traces)

| policy | Cost (sim USD / trace) | Cost red. vs reactive (%) | P95 lat (ms) | P99 lat (ms) | SLO viol. rate | Unserved frac | Scaling events/day | Underprov. frac | Mean reward | n |
|---|---|---|---|---|---|---|---|---|---|---|
| fixed | 57.17 | -38.2 | 126 | 1409 | 0.70% | 0.43% | 0.0 | 0.45% | -0.4023 | 1 |
| fixed_peak | 171.50 | -314.7 | 51 | 66 | 0.00% | 0.00% | 0.0 | 0.00% | -1.0315 | 1 |
| reactive | 41.35 | +0.0 | 193 | 1610 | 1.11% | 0.52% | 18.1 | 0.72% | -0.3426 | 1 |
| predictive | 36.86 | +10.9 | 225 | 1964 | 1.18% | 0.58% | 18.2 | 0.79% | -0.3362 | 1 |
| predictive_q | 37.67 | +8.9 | 212 | 1874 | 1.08% | 0.58% | 15.7 | 0.79% | -0.3330 | 1 |


Cost reduction is relative to the tuned reactive baseline; positive = cheaper. Read it together with SLO violations and tail latency.

- Predictive vs reactive: cost +10.9%, SLO-violating intervals 1.18% vs 1.11%, P95 225 vs 193 ms, churn 18.2 vs 18.1 events/day.

![tradeoff](../results/figures/tradeoff_all.png)
![bars](../results/figures/bars_main.png)
![timelines](../results/figures/scaling_timelines.png)

## Experiment 3 - Uncertainty-aware PPO

| policy | Cost (sim USD / trace) | Cost red. vs reactive (%) | P95 lat (ms) | P99 lat (ms) | SLO viol. rate | Unserved frac | Scaling events/day | Underprov. frac | Mean reward | n |
|---|---|---|---|---|---|---|---|---|---|---|
| reactive | 41.35 | +0.0 | 193 | 1610 | 1.11% | 0.52% | 18.1 | 0.72% | -0.3426 | 1 |
| predictive | 36.86 | +10.9 | 225 | 1964 | 1.18% | 0.58% | 18.2 | 0.79% | -0.3362 | 1 |
| predictive_q | 37.67 | +8.9 | 212 | 1874 | 1.08% | 0.58% | 15.7 | 0.79% | -0.3330 | 1 |
| ppo_point | 46.43 ± 1.41 | -12.3 ± +3.4 | 259 ± 25 | 1829 ± 388 | 1.88% ± 0.63% | 0.54% ± 0.09% | 139.5 ± 41.7 | 0.77% ± 0.05% | -0.3958 ± 0.0124 | 3 |
| ppo_full | 44.02 ± 1.03 | -6.5 ± +2.5 | 309 ± 47 | 1997 ± 310 | 2.18% ± 0.17% | 0.62% ± 0.07% | 155.9 ± 38.0 | 0.96% ± 0.06% | -0.3952 ± 0.0045 | 3 |
| ppo_base | 44.94 ± 0.31 | -8.7 ± +0.7 | 340 ± 68 | 1840 ± 180 | 2.72% ± 0.32% | 0.54% ± 0.05% | 142.3 ± 59.4 | 0.93% ± 0.09% | -0.4079 ± 0.0144 | 3 |
| ppo_unc | 44.31 ± 1.75 | -7.2 ± +4.2 | 328 ± 85 | 1813 ± 406 | 2.50% ± 0.92% | 0.56% ± 0.07% | 133.0 ± 41.3 | 0.91% ± 0.19% | -0.3991 ± 0.0193 | 3 |


Point forecast vs TFT quantiles (pending state available in both):
- **total_cost**: ppo_full 44.02 (±1.03) vs ppo_point 46.43 (±1.41) -> ppo_full is better.
- **slo_violation_rate_intervals**: ppo_full 2.18% (±0.17%) vs ppo_point 1.88% (±0.63%) -> within seed variability (no reliable difference).
- **mean_reward**: ppo_full -0.3952 (±0.0045) vs ppo_point -0.3958 (±0.0124) -> within seed variability (no reliable difference).
- **scaling_events_per_day**: ppo_full 155.9 (±38.0) vs ppo_point 139.5 (±41.7) -> within seed variability (no reliable difference).

Same comparison without pending state:
- **total_cost**: ppo_unc 44.31 (±1.75) vs ppo_base 44.94 (±0.31) -> within seed variability (no reliable difference).
- **slo_violation_rate_intervals**: ppo_unc 2.50% (±0.92%) vs ppo_base 2.72% (±0.32%) -> within seed variability (no reliable difference).
- **mean_reward**: ppo_unc -0.3991 (±0.0193) vs ppo_base -0.4079 (±0.0144) -> within seed variability (no reliable difference).

![exp3/4](../results/figures/bars_exp3_exp4.png)
![learning](../results/figures/ppo_learning_curves.png)

## Experiment 4 - Provisioning delay / pending-capacity state

| policy | Cost (sim USD / trace) | Cost red. vs reactive (%) | P95 lat (ms) | P99 lat (ms) | SLO viol. rate | Unserved frac | Scaling events/day | Underprov. frac | Mean reward | n |
|---|---|---|---|---|---|---|---|---|---|---|
| ppo_base | 44.94 ± 0.31 | -8.7 ± +0.7 | 340 ± 68 | 1840 ± 180 | 2.72% ± 0.32% | 0.54% ± 0.05% | 142.3 ± 59.4 | 0.93% ± 0.09% | -0.4079 ± 0.0144 | 3 |
| ppo_point | 46.43 ± 1.41 | -12.3 ± +3.4 | 259 ± 25 | 1829 ± 388 | 1.88% ± 0.63% | 0.54% ± 0.09% | 139.5 ± 41.7 | 0.77% ± 0.05% | -0.3958 ± 0.0124 | 3 |
| ppo_unc | 44.31 ± 1.75 | -7.2 ± +4.2 | 328 ± 85 | 1813 ± 406 | 2.50% ± 0.92% | 0.56% ± 0.07% | 133.0 ± 41.3 | 0.91% ± 0.19% | -0.3991 ± 0.0193 | 3 |
| ppo_full | 44.02 ± 1.03 | -6.5 ± +2.5 | 309 ± 47 | 1997 ± 310 | 2.18% ± 0.17% | 0.62% ± 0.07% | 155.9 ± 38.0 | 0.96% ± 0.06% | -0.3952 ± 0.0045 | 3 |


Pending state on vs off:
- **total_cost**: ppo_full 44.02 (±1.03) vs ppo_unc 44.31 (±1.75) -> within seed variability (no reliable difference).
- **total_cost**: ppo_point 46.43 (±1.41) vs ppo_base 44.94 (±0.31) -> ppo_point is worse.
- **slo_violation_rate_intervals**: ppo_full 2.18% (±0.17%) vs ppo_unc 2.50% (±0.92%) -> within seed variability (no reliable difference).
- **slo_violation_rate_intervals**: ppo_point 1.88% (±0.63%) vs ppo_base 2.72% (±0.32%) -> ppo_point is better.
- **mean_reward**: ppo_full -0.3952 (±0.0045) vs ppo_unc -0.3991 (±0.0193) -> within seed variability (no reliable difference).
- **mean_reward**: ppo_point -0.3958 (±0.0124) vs ppo_base -0.4079 (±0.0144) -> within seed variability (no reliable difference).

SLO-violating intervals (%) when the provisioning delay at test time differs from the training delay (3):

| policy     |    0 |    1 |    3 |    6 |
|:-----------|-----:|-----:|-----:|-----:|
| ppo_base   | 1.68 | 2.02 | 2.72 | 3.62 |
| ppo_full   | 1.35 | 1.54 | 2.18 | 3.24 |
| ppo_point  | 1.28 | 1.45 | 1.88 | 2.67 |
| ppo_unc    | 1.59 | 1.9  | 2.5  | 3.16 |
| predictive | 0.44 | 0.71 | 1.18 | 1.66 |
| reactive   | 0.41 | 0.61 | 1.11 | 1.67 |

![delay](../results/figures/delay_sensitivity.png)

## Experiment 5 - Anomaly signal

Detector quality on the test split (synthetic labels):

| config | precision | recall | F1 | FPR | AUROC |
|---|---|---|---|---|---|
| raw_metrics | 0.561 | 0.472 | 0.513 | 0.0104 | 0.965 |
| forecast_residuals | 0.606 | 0.536 | 0.569 | 0.0098 | 0.737 |
| union_of_both | 0.447 | 0.540 | 0.489 | 0.0188 | nan |

| policy | Cost (sim USD / trace) | Cost red. vs reactive (%) | P95 lat (ms) | P99 lat (ms) | SLO viol. rate | Unserved frac | Scaling events/day | Underprov. frac | Mean reward | n |
|---|---|---|---|---|---|---|---|---|---|---|
| ppo_full | 44.02 ± 1.03 | -6.5 ± +2.5 | 309 ± 47 | 1997 ± 310 | 2.18% ± 0.17% | 0.62% ± 0.07% | 155.9 ± 38.0 | 0.96% ± 0.06% | -0.3952 ± 0.0045 | 3 |
| ppo_full_rawz | 45.36 ± 2.36 | -9.7 ± +5.7 | 275 ± 16 | 1770 ± 340 | 2.19% ± 0.36% | 0.54% ± 0.05% | 146.7 ± 71.0 | 0.86% ± 0.08% | -0.3981 ± 0.0137 | 3 |
| ppo_full_residz | 44.97 ± 1.34 | -8.8 ± +3.2 | 279 ± 52 | 1826 ± 528 | 2.07% ± 0.80% | 0.56% ± 0.09% | 160.6 ± 53.5 | 0.89% ± 0.17% | -0.3970 ± 0.0183 | 3 |


Controller with raw-metric score / residual score vs no anomaly signal:
- **total_cost**: ppo_full_rawz 45.36 (±2.36) vs ppo_full 44.02 (±1.03) -> within seed variability (no reliable difference).
- **slo_violation_rate_intervals**: ppo_full_rawz 2.19% (±0.36%) vs ppo_full 2.18% (±0.17%) -> within seed variability (no reliable difference).
- **mean_reward**: ppo_full_rawz -0.3981 (±0.0137) vs ppo_full -0.3952 (±0.0045) -> within seed variability (no reliable difference).
- **total_cost**: ppo_full_residz 44.97 (±1.34) vs ppo_full 44.02 (±1.03) -> within seed variability (no reliable difference).
- **slo_violation_rate_intervals**: ppo_full_residz 2.07% (±0.80%) vs ppo_full 2.18% (±0.17%) -> within seed variability (no reliable difference).
- **mean_reward**: ppo_full_residz -0.3970 (±0.0183) vs ppo_full -0.3952 (±0.0045) -> within seed variability (no reliable difference).

![anomaly](../results/figures/anomaly_detection.png)
![bars exp5](../results/figures/bars_exp5.png)

## Limitations

See `docs/limitations.md`. In particular: synthetic data, simulated latency/cost, small PPO budget and few seeds.
