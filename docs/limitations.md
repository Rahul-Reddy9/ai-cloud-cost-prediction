# Limitations and safety considerations

1. **Synthetic data.** The reported experiments use a generated workload with injected anomalies. Real
   workloads have different seasonality, burstiness and failure modes; conclusions may not transfer.
2. **Simulator realism.** Capacity per replica, latency curve, billing and provisioning delay are assumptions.
   Latency is not calibrated on a load-tested application; P95/P99 are percentiles of modelled interval
   latency. Cost is simulated, not an invoice (no node packing, discounts, storage, egress).
3. **Counterfactual telemetry.** Anomaly scores from raw CPU/latency come from telemetry logged under a simple
   threshold policy; under a different controller those metrics would differ. TFT therefore uses only
   workload/calendar covariates.
4. **Forecast reuse.** TFT forecasts are pre-computed on fixed traces (demand does not react to scaling), which is
   standard for autoscaling simulations but cannot represent feedback effects such as retries or load shedding.
5. **Small PPO budget.** The sandbox run used the `quick` profile (few seeds, short training). Differences smaller
   than seed-to-seed variation should not be treated as findings; use the `full` profile for any claim.
6. **Controller interface.** Discrete actions up to +3 replicas, scale-down by 1 only; other designs are untested.
7. **Reward weights** are a documented a-priori choice; conclusions can depend on them (a sensitivity analysis is
   future work).
8. **HPA baseline.** The reactive baseline approximates HPA inside the simulator; claims about beating HPA require
   a real cluster with a properly configured HPA.
9. **Novelty not validated.** See `methodology.md` section 9.
10. **Kubernetes stage not exercised.** `deploy/` contains an observation-only scaffold and manifests that were not
    run against a cluster in this build.

## Safety considerations for any live deployment
Dedicated namespace, explicit min/max replicas, logging of every decision, readiness and pending tracking, a
conventional fallback policy, a kill switch, observation-only mode first, and budget alerts on paid clouds.
