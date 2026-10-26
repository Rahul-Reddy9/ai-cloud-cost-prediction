"""Detector evaluation on the TEST split (Part C / Experiment 5, detector half).

Thresholds were chosen on validation data. Labels come from the SYNTHETIC generator (or from controlled
synthetic injections when a dataset has no labels) and are therefore not real incidents.
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

from ..common import load_all_configs, resolve, save_json
from ..evaluation.metrics import detection_metrics
from .score_anomalies import load_detector


def inject_synthetic_anomalies(df: pd.DataFrame, seed: int = 0, rate: float = 0.02) -> pd.DataFrame:
    """Controlled anomaly scenario for datasets WITHOUT labels: multiplicative request surges /
    latency spikes on random short windows. Labels are tagged synthetic by the caller."""
    rng = np.random.default_rng(seed)
    d = df.copy()
    d["anomaly_label"] = 0
    n = len(d)
    for _ in range(int(rate * n / 5)):
        s = int(rng.integers(0, max(1, n - 6)))
        e = s + int(rng.integers(3, 7))
        if rng.random() < 0.5:
            d.iloc[s:e, d.columns.get_loc("request_rate")] *= rng.uniform(2.0, 3.0)
        else:
            d.iloc[s:e, d.columns.get_loc("latency_ms")] *= rng.uniform(4.0, 8.0)
        d.iloc[s:e, d.columns.get_loc("anomaly_label")] = 1
    return d


def evaluate_detectors(cfgs: dict | None = None) -> dict:
    cfgs = cfgs or load_all_configs()
    sc = pd.read_parquet(resolve(cfgs["anomaly"]["paths"]["scores_file"]))
    raw_df = pd.read_parquet(resolve("data/processed") / cfgs["data"]["dataset"]["processed_file"])
    if "anomaly_type" in raw_df.columns:
        sc = sc.merge(raw_df[["group_id", "time_idx", "anomaly_type"]], on=["group_id", "time_idx"], how="left")
    t = sc[(sc["split"] == "test") & sc["score_resid"].notna()]
    y = t["anomaly_label"].to_numpy()
    res = {"split": "test", "n": int(len(t)), "n_anomalous": int(y.sum()),
           "labels_are_synthetic": cfgs["data"]["dataset"]["source"] == "synthetic"}
    for name, sc_col, flag_col in (("raw_metrics", "score_raw", "flag_raw"), ("forecast_residuals", "score_resid", "flag_resid")):
        r = detection_metrics(y, t[flag_col].to_numpy())
        r["auroc"] = float(roc_auc_score(y, t[sc_col])) if 0 < y.sum() < len(y) else float("nan")
        r["average_precision"] = float(average_precision_score(y, t[sc_col])) if y.sum() > 0 else float("nan")
        if "anomaly_type" in t.columns:
            r["recall_by_type"] = {
                k: float(t.loc[t["anomaly_type"] == k, flag_col].mean())
                for k in sorted(set(t["anomaly_type"].dropna()) - {""})}
        res[name] = r
    # union of both signals (secondary reading, not tuned separately)
    union = ((t["flag_raw"] == 1) | (t["flag_resid"] == 1)).astype(int)
    res["union_of_both"] = detection_metrics(y, union)
    save_json(res, "results/metrics/detector_eval.json")
    return res


def main() -> None:
    r = evaluate_detectors()
    print(f"[detector eval] test n={r['n']} anomalous={r['n_anomalous']} (synthetic labels={r['labels_are_synthetic']})")
    for k in ("raw_metrics", "forecast_residuals", "union_of_both"):
        v = r[k]
        print(f"  {k:20s} P={v['precision']:.3f} R={v['recall']:.3f} F1={v['f1']:.3f} FPR={v['fpr']:.4f}"
              + (f" AUROC={v['auroc']:.3f}" if "auroc" in v else ""))


if __name__ == "__main__":
    main()
