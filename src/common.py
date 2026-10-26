"""Shared helpers: config loading, seeding, JSON saving, run metadata.

This is the only module added beyond the folder layout in the blueprint (plus __init__.py files).
"""
from __future__ import annotations

import hashlib
import json
import platform
import random
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any

import numpy as np
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def resolve(path: str | Path) -> Path:
    p = Path(path)
    return p if p.is_absolute() else PROJECT_ROOT / p


def load_config(name: str, config_dir: str | Path = "configs") -> dict:
    """Load configs/<name>.yaml (name without extension)."""
    path = resolve(config_dir) / f"{name}.yaml"
    with open(path, "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def load_all_configs(config_dir: str | Path = "configs") -> dict:
    return {n: load_config(n, config_dir) for n in
            ["data", "forecasting", "anomaly", "simulation", "ppo", "experiments"]}


def set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
    except Exception:  # pragma: no cover
        pass


def _jsonable(obj: Any):
    if is_dataclass(obj) and not isinstance(obj, type):
        return asdict(obj)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        v = float(obj)
        return None if not np.isfinite(v) else v
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (Path,)):
        return str(obj)
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    return str(obj)


def save_json(obj: Any, path: str | Path) -> Path:
    path = resolve(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(obj, fh, indent=2, default=_jsonable, sort_keys=True)
    return path


def load_json(path: str | Path) -> Any:
    with open(resolve(path), "r", encoding="utf-8") as fh:
        return json.load(fh)


def config_hash(cfg: Any) -> str:
    blob = json.dumps(cfg, sort_keys=True, default=_jsonable).encode()
    return hashlib.sha256(blob).hexdigest()[:12]


def environment_info() -> dict:
    info = {"python": platform.python_version(), "platform": platform.platform()}
    for mod in ["numpy", "pandas", "sklearn", "torch", "lightning", "pytorch_forecasting",
                "stable_baselines3", "gymnasium"]:
        try:
            info[mod] = __import__(mod).__version__
        except Exception:
            info[mod] = None
    return info
