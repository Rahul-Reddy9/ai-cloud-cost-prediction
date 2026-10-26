"""OPTIONAL stage 2 scaffold - observation-only controller (NOT exercised against a real cluster in this build).

It reads metrics from Prometheus, computes a recommended replica count with a conventional fallback policy and
logs every decision to JSONL. It never changes the cluster: applying changes is deliberately not implemented
(CONTROLLER_APPLY must stay false). Create a file named STOP next to this script to stop it safely.
Environment variables: see .env.example.
"""
from __future__ import annotations

import json
import os
import sys
import time
import urllib.parse
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.baselines.reactive_policy import ReactivePolicy  # noqa: E402

DELTAS = [-1, 0, 1, 3]


def prom_query(base: str, q: str) -> float:
    url = f"{base}/api/v1/query?" + urllib.parse.urlencode({"query": q})
    with urllib.request.urlopen(url, timeout=10) as r:
        data = json.load(r)
    res = data["data"]["result"]
    if not res:
        raise RuntimeError(f"no data for query {q!r}")      # missing telemetry is never treated as zero
    return float(res[0]["value"][1])


def recommend(policy, ready: int, pending: int, cpu: float, demand: float, min_r: int, max_r: int) -> int:
    """Pure decision function (unit-testable): bounded replica recommendation."""
    state = {"ready": ready, "pending": pending, "total": ready + pending, "cpu": cpu, "demand": demand, "forecast": None,
             "capacity_per_replica": 100.0, "min_replicas": min_r, "max_replicas": max_r, "deltas": DELTAS, "delay_steps": 3}
    delta = DELTAS[policy.act(state)]
    return max(min_r, min(max_r, ready + pending + delta))


def main() -> None:
    base = os.environ.get("PROMETHEUS_URL", "http://localhost:9090")
    ns, dep = os.environ.get("K8S_NAMESPACE", "autoscale-test"), os.environ.get("K8S_DEPLOYMENT", "sample-web-app")
    min_r, max_r = int(os.environ.get("MIN_REPLICAS", 2)), int(os.environ.get("MAX_REPLICAS", 12))
    if os.environ.get("CONTROLLER_APPLY", "false").lower() == "true":
        raise SystemExit("Applying changes is not implemented in this scaffold; run observation-only first.")
    stop = Path(__file__).with_name("STOP")
    log = Path(__file__).with_name("decisions.jsonl")
    policy = ReactivePolicy(target_util=0.7)
    while not stop.exists():
        try:
            ready = int(prom_query(base, f'kube_deployment_status_replicas_ready{{namespace="{ns}",deployment="{dep}"}}'))
            total = int(prom_query(base, f'kube_deployment_status_replicas{{namespace="{ns}",deployment="{dep}"}}'))
            cpu = prom_query(base, f'avg(rate(container_cpu_usage_seconds_total{{namespace="{ns}",pod=~"{dep}-.*"}}[2m])) / 0.5')
            rec = recommend(policy, ready, max(total - ready, 0), min(cpu, 1.0), 0.0, min_r, max_r)
            entry = {"ts": time.time(), "ready": ready, "total": total, "cpu": cpu, "recommended_replicas": rec, "applied": False}
        except Exception as exc:                              # log and keep the cluster untouched
            entry = {"ts": time.time(), "error": str(exc), "applied": False}
        with open(log, "a") as fh:
            fh.write(json.dumps(entry) + "\n")
        time.sleep(60)


if __name__ == "__main__":
    main()
