# Methodology

## 1. Problem
Choose, every 5 minutes, how many replicas a service should run so that **cost** is low while **latency,
SLO compliance and scaling stability** stay acceptable, given that new replicas need
`provisioning_delay` intervals (default 3 = 15 min) before they can serve traffic. The research extension
tested here is **uncertainty-aware, delay-aware predictive autoscaling**: the PPO state contains TFT forecast
uncertainty and the pending-capacity schedule. *Whether this helps is an empirical question answered by the
ablations; it is not assumed.*

## 2. Data
* **Used in this build: SYNTHETIC.** `src/environment/workload.py` generates 5 services x 40 days at 5-minute
  resolution: daily + second-harmonic seasonality, weekend factor, trend, AR(1) multiplicative noise, normal
  promotional events (not labelled) and **labelled injected anomalies** (surge, drop, latency spike, CPU
  saturation). CPU, memory, latency, replicas and cost are produced by replaying the demand through the
  simulator under a logging threshold rule. Telemetry gaps (missing timestamps / NaN) are injected on purpose.
* **Real data:** put it in `data/raw/`, set `dataset.source: csv`, `raw_file` and `column_map` in
  `configs/data.yaml`. Missing schema columns are reported by validation; `resource_cost` is derived from an
  explicit price model (currency, price, billing interval and assumptions recorded in `preprocess_config.json`).
  Candidate public traces (verify licence and format yourself): Microsoft Azure Public Dataset, Alibaba cluster
  trace, Google cluster-data.
* **Cleaning rules:** sort, de-duplicate (keep last), regular grid, linear interpolation only for *complete*
  gaps of at most 6 steps, longer gaps remove the rows and start a new segment (windows never cross a segment
  boundary). Missing values are never replaced by zero.
* **Split:** chronological 70/15/15 on the shared time grid; scalers (mean/std) fitted on the training rows only.

## 3. Forecasting (TFT)
`pytorch-forecasting` Temporal Fusion Transformer, encoder 48 steps (4 h), horizon 12 steps (1 h), quantiles
{0.1, 0.5, 0.9}, quantile loss, per-segment softplus target normalisation fitted on training data,
early stopping on validation loss. Covariates are only **workload and calendar features** (known:
hour/day-of-week sin-cos; unknown: request rate and its causal rolling mean). CPU / latency are excluded because
they depend on the replica count chosen by whichever controller is running; a forecast made offline from them
would not exist under a new controller. Quantile crossings are removed by sorting.
The horizon 12 is a design choice: it covers the provisioning delay (3) plus a margin; it is not claimed optimal.

Baselines on identical windows: persistence, 12-step moving average, seasonal-naive (same time yesterday).

## 4. Anomaly detection (Isolation Forest)
Two configurations: **raw** (request rate, CPU, memory, latency, standardised with train statistics) and
**residual** (one-step TFT residual scaled by the training std and by the forecast interval width). Fit data are
normal observations only: raw -> training split, residual -> first half of validation (training residuals are
in-sample). Threshold: maximum F1 on validation rows not used for fitting; fallback quantile when no labels.
Scores are normalised to [0, 2] with quantiles of the fit scores. The test split is never used for fitting or
thresholding. Labels are synthetic and results must be described as such.
For the controller, scores are pre-computed from the *logged* telemetry timeline; the residual score uses the
forecast made at `t-1` and the actual at `t`, so both are available at decision time. Limitation: the raw score
uses telemetry produced by the logging policy rather than by the policy under test.

## 5. Simulator
* **Capacity:** `min(CPU-bound, memory-bound)` requests/s per replica (CPU 2 cores / 20 ms per request = 100 rps;
  memory limit 140 rps) times ready replicas.
* **Provisioning:** a scale-up request with delay `d` is billed immediately and serves after `d` full intervals;
  scale-down cancels pending replicas first, then terminates ready replicas.
* **Cost:** `C = sum_t n_t * p * dt` with billable replicas `n_t` (ready + pending), price `p = 0.05 USD /
  replica-hour` (assumption). Simulated cost is **not** a cloud invoice.
* **Latency (synthetic approximation):** `L = L0 / (1 - rho)` for `rho < 0.95`, then linear growth, capped at a
  timeout; `rho = demand / ready capacity`. An interval violates the SLO when `L > 300 ms` or requests are dropped.
  P95/P99 are request-weighted percentiles over intervals of this modelled latency, not per-request measurements.
* **Noise:** small Gaussian noise on observed CPU/memory; optional provisioning-delay jitter (off by default).

## 6. Controller
* **Actions:** `{-1, 0, +1, +3}` replicas (discrete).
* **State:** CPU, memory, observed demand, hour sin/cos, ready replicas, forecast medians at horizons {1,3,6,12},
  [uncertainty: 10-90 % widths at the same horizons], [pending replicas by remaining delay], a cost state and
  [anomaly score]. Values are normalised by maximum capacity. Without pending information the cost state uses ready
  replicas only, so pending capacity cannot leak through the cost signal.
* **Reward:** `r = -(a*C + b*L + g*V + d*H)` with `C` cost / max cost, `L` min(latency/SLO, 3), `V` 1[violation] +
  dropped fraction, `H` |replica change| / max step; weights `a=1, b=0.2, g=1.5, d=0.05` (documented in
  `configs/simulation.yaml`; chosen a priori, **not tuned on test**).
* **Safety:** every action - PPO or baseline - passes the shield (invalid ids, min/max replicas, scale-down
  feasibility). Reward penalties alone are not treated as safety.
* **PPO:** Stable-Baselines3, MLP 64x64. At evaluation actions are *sampled* from the learned policy with a fixed RNG seed (argmax evaluation collapsed to the no-op action when probabilities were split, see `PPOPolicy`). Training episodes (288 steps) are sampled from
  the first 70 % of the validation period (TFT forecasts there are out-of-sample); the checkpoint with the best
  mean reward on the last 30 % of validation is kept. Test traces are never used for training or selection.

## 7. Baselines
Fixed (sized from the 95th percentile of *training* demand; and a peak-sized variant), reactive HPA-like
(`ceil(ready * cpu / target)`, tolerance band, scale-down stabilisation window), predictive (scale for the
forecast demand when new replicas become ready; median version and 90 %-quantile version). Reactive and
predictive hyper-parameters are tuned on the validation traces with the same reward, then frozen.
The reactive baseline is a simulator approximation of HPA, not a complete HPA.

## 8. Evaluation protocol
All policies run on the same test traces (the 5 services' test periods) with the same constraints and the same
safety shield. PPO is trained with several seeds; means and standard deviations (ddof=1) over seeds are
reported, and every seed is kept in `results/metrics/all_runs.csv`. Cost reduction is
`100 (C_baseline - C_proposed) / C_baseline` against the named baseline (reactive) and is always shown next to
SLO violations, latency, churn and under-provisioning.

## 9. Literature positioning (validation still required)
Components follow published methods: Temporal Fusion Transformer (Lim, Arik, Loeff, Pfister, *Int. J. of
Forecasting*, 2021), PPO (Schulman et al., arXiv:1707.06347, 2017), Isolation Forest (Liu, Ting, Zhou, ICDM 2008).
The Kubernetes HPA algorithm is documented in the Kubernetes documentation. **No novelty claim is made.** Before
submitting a thesis, run a structured literature search (e.g. "predictive autoscaling", "uncertainty-aware
autoscaling", "delay-aware / provisioning-delay reinforcement learning autoscaling", "TFT workload forecasting")
and record related work and differences; the references above should be re-checked against the originals.
