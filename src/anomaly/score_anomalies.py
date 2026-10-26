"""Score every observation with the trained Isolation Forests and save timestamps + scores (Part C)."""
from __future__ import annotations

import json

import joblib
import numpy as np
import pandas as pd

from ..common import load_all_configs, resolve
from .train_detector import anomaly_score, normalise, raw_features, residual_table


def load_detector(cfgs: dict, name: str) -> dict:
    return joblib.load(resolve(cfgs["anomaly"]["paths"]["model_dir"]) / f"{name}_detector.joblib")


def score_frame(cfgs: dict, df: pd.DataFrame, predictions: pd.DataFrame | None, meta: dict) -> pd.DataFrame:
    """Return df-aligned scores. z_* are normalised to [0, 2]; residual scores exist only where a
    1-step forecast is available (validation/test)."""
    scalers = meta["scalers_train_only"]
    raw = load_detector(cfgs, "raw")
    out = df[["group_id", "service_id", "time_idx", "timestamp", "split"] +
             (["anomaly_label"] if "anomaly_label" in df.columns else [])].copy()
    s = anomaly_score(raw["model"], raw_features(df, scalers, raw["features"]))
    out["score_raw"], out["z_raw"] = s, normalise(s, raw["norm"])
    out["flag_raw"] = (s >= raw["threshold"]).astype(int)
    out["score_resid"], out["z_resid"], out["flag_resid"] = np.nan, np.nan, 0
    if predictions is not None:
        res = load_detector(cfgs, "residual")
        rt = residual_table(df, predictions, scalers)
        sr = anomaly_score(res["model"], rt[res["features"]].to_numpy())
        rt["score_resid"], rt["z_resid"] = sr, normalise(sr, res["norm"])
        rt["flag_resid"] = (sr >= res["threshold"]).astype(int)
        out = out.drop(columns=["score_resid", "z_resid", "flag_resid"]).merge(
            rt[["group_id", "time_idx", "score_resid", "z_resid", "flag_resid"]],
            on=["group_id", "time_idx"], how="left")
        out["flag_resid"] = out["flag_resid"].fillna(0).astype(int)
    return out


def score_and_save(cfgs: dict, df: pd.DataFrame, predictions: pd.DataFrame, meta: dict) -> pd.DataFrame:
    out = score_frame(cfgs, df, predictions, meta)
    path = resolve(cfgs["anomaly"]["paths"]["scores_file"])
    path.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(path, index=False)
    print(f"[anomaly] wrote {len(out)} scored rows -> {path}")
    return out


def main() -> None:
    cfgs = load_all_configs()
    from ..data.load_data import load_processed
    df = load_processed(cfgs["data"])
    meta = json.load(open(resolve("data/processed/preprocess_config.json")))
    pred = pd.read_parquet(resolve(cfgs["forecasting"]["paths"]["predictions_file"]))
    score_and_save(cfgs, df, pred, meta)


if __name__ == "__main__":
    main()
