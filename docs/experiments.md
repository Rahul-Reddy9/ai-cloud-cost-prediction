# Experiments

All policies are evaluated on the **same held-out test traces** (the test period of each of the 5 services),
with identical resource limits, safety shield, provisioning delay and seeds for the simulator noise.
Baseline hyper-parameters are tuned on the validation traces only. PPO is trained with several seeds; every
seed is reported (`results/metrics/all_runs.csv`), tables show mean ± sample std (ddof = 1).

| # | Question | Policies compared | Output |
|---|---|---|---|
| 1 | Does TFT beat simple forecasts? | TFT median vs persistence, moving average, seasonal-naive | `forecast_eval.json`, `forecast_*.png` |
| 2 | Does predictive scaling beat tuned reactive scaling? | fixed, fixed-peak, reactive, predictive (median / upper quantile) | `table_exp2_baselines.*` |
| 3 | Does TFT uncertainty help PPO? | `ppo_point` vs `ppo_full`; `ppo_base` vs `ppo_unc` | `table_exp3_uncertainty.*` |
| 4 | Does pending-capacity state help under provisioning delay? | `ppo_unc` vs `ppo_full`; `ppo_base` vs `ppo_point`; delay sweep 0/1/3/6 | `table_exp4_*` , `delay_sensitivity.png` |
| 5 | Does an anomaly signal add value? | `ppo_full` (none) vs `+raw` vs `+residual` | `table_exp5_anomaly.*`, `detector_eval.json` |

Variant definitions (`configs/ppo.yaml`):

| variant | uncertainty (q10-q90) | pending state | anomaly score |
|---|---|---|---|
| ppo_base | no | no | none |
| ppo_point | no | yes | none |
| ppo_unc | yes | no | none |
| ppo_full | yes | yes | none |
| ppo_full_rawz | yes | yes | raw-metric Isolation Forest |
| ppo_full_residz | yes | yes | forecast-residual Isolation Forest |

## Metrics
* **Total cost** = sum over intervals of billable replicas x price x dt (simulated USD per trace).
* **Cost reduction (%)** = 100 (C_baseline - C_proposed) / C_baseline, baseline = tuned reactive, computed per run.
* **Forecast** MAE / RMSE, mean pinball loss over (0.1, 0.5, 0.9), 10-90 % interval coverage and width.
* **P95 / P99 latency**: request-weighted percentile over intervals of the simulated latency.
* **SLO violation rate**: fraction of intervals (and of requests) with modelled latency > 300 ms or dropped requests.
* **Scaling churn**: scaling events per day (and total replicas changed).
* **Under-provisioning**: intervals where demand exceeds ready capacity (rho > 1).
* **Detector** precision / recall / F1 / FPR / AUROC on the test split (thresholds from validation).

## Reproduce
```bash
python scripts/prepare_data.py --regenerate
python scripts/train_all.py --profile quick        # TFT, Isolation Forest, PPO variants x seeds
python scripts/run_all_experiments.py --profile quick
streamlit run src/dashboard/app.py
```
`--profile full` uses 5 seeds and 300 000 PPO steps (see `configs/experiments.yaml`). Each run stores its
configuration snapshot, hashes, seed and library versions in `results/metrics/runs/<run_id>.json`.
