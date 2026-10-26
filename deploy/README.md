# Optional stage 2: Kubernetes testbed (scaffold, not run in this build)

Order of work: (1) get the simulator results first; (2) deploy `k8s/*.yaml` to a local kind/minikube cluster or a
dedicated test namespace; (3) install Prometheus + kube-state-metrics; (4) run `observe_only.py` and compare its
recommendations with the real HPA (`20-hpa-baseline.yaml`) before allowing any automated change.

Safety checklist: dedicated namespace, explicit min/max replicas, every decision logged (`decisions.jsonl`),
readiness / pending tracking, a conventional fallback policy (the reactive rule used here), a `STOP` file kill
switch, and budget alerts if you use a paid cloud. Applying changes is intentionally not implemented.
