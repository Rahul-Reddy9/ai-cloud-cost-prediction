"""Cleaning, resampling, chronological splitting and feature engineering (Part A).

Rules: (1) timestamps validated and sorted; (2) regular grid; (3) missing telemetry is interpolated only
for short gaps and is NEVER treated as zero utilisation - longer gaps remove the rows and start a new
segment so no window crosses the gap; (4) scalers fit on the training partition only; (5) chronological
train/validation/test split; (6) all settings are saved for reproducibility.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..common import config_hash, resolve, save_json
from ..environment.cost_model import PriceModel
from .validate_data import validate_dataframe

NUMERIC_COLS = ["request_rate", "cpu_utilization", "memory_utilization", "latency_ms",
                "replica_count", "pending_replicas", "resource_cost"]
COUNT_COLS = ["replica_count", "pending_replicas"]


def _short_gap_mask(na: pd.Series, max_gap: int) -> pd.Series:
    """True for NaN positions that belong to a NaN run of length <= max_gap (longer runs stay missing)."""
    run_id = (na != na.shift()).cumsum()
    run_len = na.groupby(run_id).transform("size")
    return na & (run_len <= max_gap)


def _regularise(g: pd.DataFrame, grid: pd.DatetimeIndex, max_gap: int, cols: list[str]) -> pd.DataFrame:
    g = g.sort_values("timestamp").drop_duplicates("timestamp", keep="last").set_index("timestamp")
    g = g.reindex(grid)
    g["imputed"] = False
    for c in cols:
        na = g[c].isna()
        if not na.any():
            continue
        if c in COUNT_COLS:
            full = g[c].ffill()
        else:
            full = g[c].interpolate(method="linear", limit_area="inside")
        fill = _short_gap_mask(na, max_gap) & full.notna()      # only complete short gaps are imputed
        g.loc[fill, c] = full[fill]
        g["imputed"] |= fill
    if "anomaly_label" in g.columns:
        g["anomaly_label"] = g["anomaly_label"].fillna(0).astype(int)   # unlabeled rows: assume normal
    return g


def preprocess(df: pd.DataFrame, data_cfg: dict, price_cfg: dict | None = None):
    """Return (processed_df, meta). `price_cfg` is only used when resource_cost is absent."""
    report = validate_dataframe(df, data_cfg)
    report.raise_if_errors()
    interval = int(data_cfg["dataset"]["interval_minutes"])
    max_gap = int(data_cfg["cleaning"]["max_gap_steps"])
    df = df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    if "service_id" not in df.columns:
        df["service_id"] = "svc1"

    cols = [c for c in NUMERIC_COLS if c in df.columns]
    pm_cfg = data_cfg["price_model"]
    price = PriceModel(float(pm_cfg["price_per_replica_hour"]), float(pm_cfg["billing_interval_minutes"]),
                       pm_cfg["currency"])
    derived_cost = False
    if "resource_cost" not in df.columns and "replica_count" in df.columns:
        pend = df["pending_replicas"] if "pending_replicas" in df.columns else 0
        df["resource_cost"] = (df["replica_count"] + pend) * price.price_per_replica_hour * price.dt_hours
        cols.append("resource_cost")
        derived_cost = True

    grid = pd.date_range(df["timestamp"].min().floor(f"{interval}min"),
                         df["timestamp"].max().ceil(f"{interval}min"), freq=f"{interval}min")
    n_grid = len(grid)
    n_train = int(round(n_grid * data_cfg["split"]["train"]))
    n_val = int(round(n_grid * data_cfg["split"]["validation"]))
    split_arr = np.array(["train"] * n_train + ["validation"] * n_val + ["test"] * (n_grid - n_train - n_val))

    parts, dropped = [], 0
    for sid, g in df.groupby("service_id", sort=True):
        keep_cols = ["timestamp"] + [c for c in cols + ["anomaly_label", "anomaly_type"] if c in g.columns]
        r = _regularise(g[keep_cols], grid, max_gap, cols)
        r["time_idx"] = np.arange(n_grid)
        r["split"] = split_arr
        invalid = ~np.isfinite(r[cols].to_numpy(dtype=float)).all(axis=1)
        seg = np.cumsum(invalid)
        r["segment"] = seg
        dropped += int(invalid.sum())
        r = r.loc[~invalid]
        r["service_id"] = sid
        r["group_id"] = [f"{sid}_s{s}" for s in r["segment"]]
        r.index.name = "timestamp"
        parts.append(r.reset_index())
    out = pd.concat(parts, ignore_index=True)
    out = out.sort_values(["group_id", "time_idx"]).reset_index(drop=True)

    # causal features
    hour = out["timestamp"].dt.hour + out["timestamp"].dt.minute / 60.0
    dow = out["timestamp"].dt.dayofweek
    out["hour_sin"], out["hour_cos"] = np.sin(2 * np.pi * hour / 24), np.cos(2 * np.pi * hour / 24)
    out["dow_sin"], out["dow_cos"] = np.sin(2 * np.pi * dow / 7), np.cos(2 * np.pi * dow / 7)
    w = int(data_cfg["features"]["rolling_window_steps"])
    out["request_rate_roll"] = out.groupby("group_id")["request_rate"].transform(
        lambda s: s.rolling(w, min_periods=1).mean())
    out["request_rate_diff"] = out.groupby("group_id")["request_rate"].diff().fillna(0.0)

    # scalers: TRAIN partition only
    train_rows = out["split"] == "train"
    scalers = {}
    for c in data_cfg["features"]["scale_columns"]:
        if c in out.columns:
            s = out.loc[train_rows, c]
            scalers[c] = {"mean": float(s.mean()), "std": float(s.std(ddof=0) or 1.0)}

    boundaries = {
        "train_end": str(grid[n_train - 1]), "validation_start": str(grid[n_train]),
        "validation_end": str(grid[n_train + n_val - 1]), "test_start": str(grid[n_train + n_val]),
        "test_end": str(grid[-1]),
        "train_end_idx": n_train - 1, "validation_start_idx": n_train,
        "validation_end_idx": n_train + n_val - 1, "test_start_idx": n_train + n_val,
        "test_end_idx": n_grid - 1, "n_grid": n_grid,
    }
    meta = {
        "interval_minutes": interval, "validation_report": report.to_dict(), "boundaries": boundaries,
        "scalers_train_only": scalers, "rows_dropped_long_gaps": dropped,
        "rows_imputed": int(out["imputed"].sum()), "n_groups": int(out["group_id"].nunique()),
        "cost_derived_from_price_model": derived_cost,
        "price_model": pm_cfg, "cleaning": data_cfg["cleaning"], "split": data_cfg["split"],
        "data_config_hash": config_hash(data_cfg),
        "rows_per_split": {k: int(v) for k, v in out["split"].value_counts().items()},
    }
    return out, meta


def save_processed(out: pd.DataFrame, meta: dict, data_cfg: dict) -> None:
    p = resolve("data/processed") / data_cfg["dataset"]["processed_file"]
    p.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(p, index=False)
    save_json(meta, resolve("data/processed") / "preprocess_config.json")
