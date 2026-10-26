"""Dataset loading (Part A).

Supports (a) a labelled SYNTHETIC multi-service workload generated offline and (b) any real trace in
CSV/Parquet form mapped onto the internal schema with `dataset.column_map`.
Internal schema: timestamp, request_rate, cpu_utilization, memory_utilization, latency_ms,
replica_count, pending_replicas, resource_cost, anomaly_label (+ service_id).
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from ..common import load_all_configs, resolve, save_json

SCHEMA_COLUMNS = ["timestamp", "cpu_utilization", "memory_utilization", "request_rate", "latency_ms",
                  "replica_count", "pending_replicas", "resource_cost", "anomaly_label"]


def raw_path(cfg: dict) -> Path:
    return resolve("data/raw") / cfg["dataset"]["raw_file"]


def processed_path(cfg: dict) -> Path:
    return resolve("data/processed") / cfg["dataset"]["processed_file"]


def read_table(path: Path) -> pd.DataFrame:
    if path.suffix == ".parquet":
        return pd.read_parquet(path)
    return pd.read_csv(path)


def load_raw(cfg: dict | None = None, sim_cfg: dict | None = None, regenerate: bool = False) -> pd.DataFrame:
    """Load data/raw/<raw_file>; generate the synthetic dataset first when needed."""
    if cfg is None or sim_cfg is None:
        allc = load_all_configs()
        cfg, sim_cfg = allc["data"], allc["simulation"]
    path = raw_path(cfg)
    ds = cfg["dataset"]
    if ds["source"] == "synthetic" and (regenerate or not path.exists()):
        from ..environment.workload import generate_dataset
        df = generate_dataset(cfg, sim_cfg)
        path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(path, index=False)
        save_json({
            "name": ds["name"], "synthetic": True,
            "note": "SYNTHETIC workload with injected labelled anomalies; not a real production trace.",
            "n_rows": len(df), "generator_config": cfg["synthetic"],
        }, path.parent / "DATASET_INFO.json")
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Put the original dataset in data/raw/ and set dataset.raw_file / column_map "
            f"in configs/data.yaml, or use dataset.source: synthetic.")
    df = read_table(path)
    if ds.get("column_map"):
        df = df.rename(columns=ds["column_map"])
    svc = ds.get("service_column", "service_id")
    if svc != "service_id" and svc in df.columns:
        df = df.rename(columns={svc: "service_id"})
    if "service_id" not in df.columns:
        df["service_id"] = "svc1"
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    return df


def load_processed(cfg: dict | None = None) -> pd.DataFrame:
    cfg = cfg or load_all_configs()["data"]
    p = processed_path(cfg)
    if not p.exists():
        raise FileNotFoundError(f"{p} not found; run `python scripts/prepare_data.py` first.")
    return pd.read_parquet(p)
