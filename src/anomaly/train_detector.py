"""Isolation Forest training (Part C).

Two configurations are trained and compared:
  raw      : CPU, memory, request rate, latency (standardised with TRAIN-only statistics)
  residual : forecast residuals of the TFT one-step-ahead median (requires forecasts)

Fit data: normal observations only (anomaly_label == 0 when labels exist).  The raw detector is fitted
on the training split; the residual detector on the FIRST half of the validation split (training-period
residuals are in-sample and optimistic).  Decision thresholds are selected on validation data not used
for fitting.  The test split is never used for fitting or threshold selection.
"""
from __future__ import annotations

import json

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest

from ..common import load_all_configs, resolve, save_json
from ..evaluation.metrics import detection_metrics


# ----------------------------------------------------------------------------------------------
def raw_features(df: pd.DataFrame, scalers: dict, features: list[str]) -> np.ndarray:
    cols = []
    for f in features:
        s = scalers[f]
        cols.append((df[f].to_numpy(float) - s["mean"]) / s["std"])
    return np.stack(cols, axis=1)


def residual_table(df: pd.DataFrame, predictions: pd.DataFrame, scalers: dict) -> pd.DataFrame:
    """Per (group, time_idx): residual of the 1-step-ahead forecast issued at the previous step.

    Available at decision time t because the forecast was made at t-1 and the actual at t is observed.
    """
    p1 = predictions[predictions["horizon_step"] == 1][
        ["group_id", "target_time_idx", "median", "lower", "upper"]].rename(
        columns={"target_time_idx": "time_idx"})
    m = df[["group_id", "time_idx", "request_rate"]].merge(p1, on=["group_id", "time_idx"], how="inner")
    resid = m["request_rate"] - m["median"]
    width = np.maximum(m["upper"] - m["lower"], 0.05 * m["median"].abs() + 1.0)
    m["resid_scaled"] = resid / scalers["request_rate"]["std"]
    m["resid_z"] = np.clip(resid / width, -10, 10)
    return m[["group_id", "time_idx", "resid_scaled", "resid_z"]]


def _fit_if(X: np.ndarray, icfg: dict) -> IsolationForest:
    return IsolationForest(n_estimators=int(icfg["n_estimators"]), max_samples=min(int(icfg["max_samples"]), len(X)),
                           contamination=icfg["contamination"], random_state=int(icfg["random_state"])).fit(X)


def anomaly_score(model: IsolationForest, X: np.ndarray) -> np.ndarray:
    """Higher = more anomalous (negated sklearn score_samples)."""
    return -model.score_samples(X)


def normalise(score: np.ndarray, norm: dict) -> np.ndarray:
    return np.clip((score - norm["lo"]) / max(norm["hi"] - norm["lo"], 1e-9), 0.0, 2.0)


def pick_threshold(score: np.ndarray, labels: np.ndarray | None, fallback_q: float) -> tuple[float, str]:
    if labels is not None and labels.sum() > 0 and (labels == 0).sum() > 0:
        cands = np.unique(np.quantile(score, np.linspace(0.80, 0.999, 120)))
        best = max(cands, key=lambda t: detection_metrics(labels, score >= t)["f1"])
        return float(best), "max_f1_on_validation"
    return float(np.quantile(score, fallback_q)), f"quantile_{fallback_q}"


def train_detectors(cfgs: dict, df: pd.DataFrame, predictions: pd.DataFrame, meta: dict) -> dict:
    acfg = cfgs["anomaly"]
    scalers = meta["scalers_train_only"]
    has_labels = "anomaly_label" in df.columns and df["anomaly_label"].sum() > 0
    normal = (df["anomaly_label"] == 0) if has_labels else pd.Series(True, index=df.index)
    out_dir = resolve(acfg["paths"]["model_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    info = {"has_labels": bool(has_labels), "labels_are_synthetic": cfgs["data"]["dataset"]["source"] == "synthetic"}

    # ---- raw detector ---------------------------------------------------------------------------
    feats = list(acfg["raw_features"])
    tr = df[(df["split"] == "train") & normal]
    Xtr = raw_features(tr, scalers, feats)
    raw_if = _fit_if(Xtr, acfg["isolation_forest"])
    s_fit = anomaly_score(raw_if, Xtr)
    qn = acfg["score_normalisation"]
    raw_norm = {"lo": float(np.quantile(s_fit, qn["low_quantile"])), "hi": float(np.quantile(s_fit, qn["high_quantile"]))}
    va = df[df["split"] == "validation"]
    s_val = anomaly_score(raw_if, raw_features(va, scalers, feats))
    thr_raw, how = pick_threshold(s_val, va["anomaly_label"].to_numpy() if has_labels else None,
                                  acfg["threshold"]["fallback_quantile"])
    joblib.dump({"model": raw_if, "features": feats, "norm": raw_norm, "threshold": thr_raw}, out_dir / "raw_detector.joblib")
    info["raw"] = {"n_fit": int(len(Xtr)), "threshold": thr_raw, "threshold_method": how, "norm": raw_norm}

    # ---- residual detector ----------------------------------------------------------------------
    rt = residual_table(df, predictions, scalers)
    d = df.merge(rt, on=["group_id", "time_idx"], how="inner")
    val = d[d["split"] == "validation"].sort_values("time_idx")
    cut = val["time_idx"].min() + (val["time_idx"].max() - val["time_idx"].min()) // 2
    fit_part = val[(val["time_idx"] <= cut) & ((val["anomaly_label"] == 0) if has_labels else True)]
    thr_part = val[val["time_idx"] > cut]
    rf = list(acfg["residual_features"])
    res_if = _fit_if(fit_part[rf].to_numpy(), acfg["isolation_forest"])
    s_fit_r = anomaly_score(res_if, fit_part[rf].to_numpy())
    res_norm = {"lo": float(np.quantile(s_fit_r, qn["low_quantile"])), "hi": float(np.quantile(s_fit_r, qn["high_quantile"]))}
    s_thr = anomaly_score(res_if, thr_part[rf].to_numpy())
    thr_res, how_r = pick_threshold(s_thr, thr_part["anomaly_label"].to_numpy() if has_labels else None,
                                    acfg["threshold"]["fallback_quantile"])
    joblib.dump({"model": res_if, "features": rf, "norm": res_norm, "threshold": thr_res, "fit_until_time_idx": int(cut)},
                out_dir / "residual_detector.joblib")
    info["residual"] = {"n_fit": int(len(fit_part)), "threshold": thr_res, "threshold_method": how_r,
                        "norm": res_norm, "fit_until_time_idx": int(cut)}
    save_json(info, out_dir / "detector_info.json")
    return info


def main() -> None:
    cfgs = load_all_configs()
    from ..data.load_data import load_processed
    df = load_processed(cfgs["data"])
    meta = json.load(open(resolve("data/processed/preprocess_config.json")))
    pred = pd.read_parquet(resolve(cfgs["forecasting"]["paths"]["predictions_file"]))
    info = train_detectors(cfgs, df, pred, meta)
    print("[anomaly] raw:", info["raw"]["threshold_method"], " residual:", info["residual"]["threshold_method"])
    from .score_anomalies import score_and_save
    score_and_save(cfgs, df, pred, meta)


if __name__ == "__main__":
    main()
