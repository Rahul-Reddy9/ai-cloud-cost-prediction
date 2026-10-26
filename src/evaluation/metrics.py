"""Metric definitions (Part G): forecasting, detection and closed-loop system metrics."""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..environment.cost_model import cost_reduction_pct  # re-exported


# ----------------------------------------------------------------------------------------------
# Forecasting
# ----------------------------------------------------------------------------------------------
def mae(y, yhat) -> float:
    return float(np.mean(np.abs(np.asarray(y) - np.asarray(yhat))))


def rmse(y, yhat) -> float:
    return float(np.sqrt(np.mean((np.asarray(y) - np.asarray(yhat)) ** 2)))


def pinball_loss(y, qhat, q: float) -> float:
    """Quantile (pinball) loss for quantile level q."""
    d = np.asarray(y) - np.asarray(qhat)
    return float(np.mean(np.maximum(q * d, (q - 1) * d)))


def mean_quantile_loss(y, quantile_preds: dict) -> float:
    """Average pinball loss over {q: prediction}."""
    return float(np.mean([pinball_loss(y, p, q) for q, p in quantile_preds.items()]))


def interval_coverage(y, lower, upper) -> float:
    y, lower, upper = map(np.asarray, (y, lower, upper))
    return float(np.mean((y >= lower) & (y <= upper)))


def interval_width(lower, upper) -> float:
    return float(np.mean(np.asarray(upper) - np.asarray(lower)))


# ----------------------------------------------------------------------------------------------
# Detection
# ----------------------------------------------------------------------------------------------
def detection_metrics(labels, flags) -> dict:
    """Precision / recall / F1 / false-positive rate for binary flags vs ground-truth labels."""
    y = np.asarray(labels).astype(int)
    p = np.asarray(flags).astype(int)
    tp = int(((y == 1) & (p == 1)).sum())
    fp = int(((y == 0) & (p == 1)).sum())
    fn = int(((y == 1) & (p == 0)).sum())
    tn = int(((y == 0) & (p == 0)).sum())
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    fpr = fp / (fp + tn) if fp + tn else 0.0
    return {"precision": prec, "recall": rec, "f1": f1, "fpr": fpr, "tp": tp, "fp": fp, "fn": fn, "tn": tn}


# ----------------------------------------------------------------------------------------------
# Closed-loop (controller) metrics
# ----------------------------------------------------------------------------------------------
def weighted_quantile(values, weights, q: float) -> float:
    v = np.asarray(values, float)
    w = np.asarray(weights, float)
    if len(v) == 0 or w.sum() <= 0:
        return float("nan")
    order = np.argsort(v)
    v, w = v[order], w[order]
    cum = (np.cumsum(w) - 0.5 * w) / w.sum()
    return float(np.interp(q, cum, v))


def episode_metrics(steps: pd.DataFrame, interval_minutes: float = 5.0) -> dict:
    """Metrics of one policy run on one trace from its per-step log.

    Latency percentiles are request-weighted percentiles over intervals of the SIMULATED interval
    latency (not per-request measurements).  An interval violates the SLO when its modelled latency
    exceeds the target or any requests were dropped.
    """
    n = len(steps)
    if n == 0:
        raise ValueError("empty step log")
    demand = steps["demand"].to_numpy(float)
    viol = steps["violated"].to_numpy(bool)
    days = n * interval_minutes / (60 * 24)
    applied = steps["applied_delta"].to_numpy(float)
    return {
        "n_steps": int(n),
        "total_cost": float(steps["cost"].sum()),
        "mean_replicas_billable": float(steps["billable"].mean()),
        "mean_replicas_ready": float(steps["ready"].mean()),
        "slo_violation_rate_intervals": float(viol.mean()),
        "slo_violation_rate_requests": float((demand * viol).sum() / max(demand.sum(), 1e-9)),
        "p95_latency_ms": weighted_quantile(steps["latency_ms"], demand, 0.95),
        "p99_latency_ms": weighted_quantile(steps["latency_ms"], demand, 0.99),
        "mean_latency_ms": float(np.average(steps["latency_ms"], weights=np.maximum(demand, 1e-9))),
        "unserved_fraction": float(steps["dropped"].sum() / max(demand.sum(), 1e-9)),
        "scaling_events": int((applied != 0).sum()),
        "replicas_changed": float(np.abs(applied).sum()),
        "scaling_events_per_day": float((applied != 0).sum() / days),
        "underprovisioning_minutes": float(steps["under_provisioned"].sum() * interval_minutes),
        "underprovisioning_fraction": float(steps["under_provisioned"].mean()),
        "mean_reward": float(steps["reward"].mean()) if "reward" in steps else float("nan"),
        "total_reward": float(steps["reward"].sum()) if "reward" in steps else float("nan"),
        "safety_interventions": int(steps["safety_intervened"].sum()) if "safety_intervened" in steps else 0,
    }


def aggregate_traces(per_trace: list[dict]) -> dict:
    """Policy-level metrics = mean over traces (costs are also summed)."""
    df = pd.DataFrame(per_trace)
    agg = df.mean(numeric_only=True).to_dict()
    agg["total_cost_sum"] = float(df["total_cost"].sum())
    return agg


def summarise_seeds(runs: list[dict], keys: list[str] | None = None) -> dict:
    """mean / std (ddof=1) / min / max across seeds for each metric."""
    df = pd.DataFrame(runs)
    keys = keys or [c for c in df.columns if df[c].dtype.kind in "fi"]
    out = {}
    for k in keys:
        v = df[k].to_numpy(float)
        out[k] = {"mean": float(np.mean(v)), "std": float(np.std(v, ddof=1)) if len(v) > 1 else 0.0,
                  "min": float(v.min()), "max": float(v.max()), "n": int(len(v))}
    return out
