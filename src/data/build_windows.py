"""Historical forecasting windows (Part A) with explicit leakage guards."""
from __future__ import annotations

import numpy as np
import pandas as pd


def assert_chronological_split(df: pd.DataFrame) -> None:
    """Every train timestamp precedes every validation timestamp, which precedes every test timestamp."""
    tmax = df.groupby("split")["time_idx"].max()
    tmin = df.groupby("split")["time_idx"].min()
    if "train" in tmax and "validation" in tmin:
        assert tmax["train"] < tmin["validation"], "train overlaps validation"
    if "validation" in tmax and "test" in tmin:
        assert tmax["validation"] < tmin["test"], "validation overlaps test"


def make_windows(df: pd.DataFrame, target: str, encoder_len: int, horizon: int, split: str,
                 feature_cols: list[str] | None = None, allow_past_context: bool = True) -> dict:
    """Build sliding windows whose TARGETS all lie inside `split`.

    allow_past_context=True lets the encoder reach back into earlier splits (legitimate for validation/
    test forecasting); False requires the whole window to be inside `split` (use for training).
    A window never spans a gap or a segment boundary.  Returns arrays X (N, enc, F), y (N, H) and the
    origin / target time indices.
    """
    feature_cols = feature_cols or [target]
    Xs, ys, origins, groups, last_enc_idx = [], [], [], [], []
    for gid, g in df.sort_values(["group_id", "time_idx"]).groupby("group_id"):
        t = g["time_idx"].to_numpy()
        if len(t) < encoder_len + horizon:
            continue
        contiguous = np.diff(t) == 1
        vals = g[feature_cols].to_numpy(dtype=float)
        tgt = g[target].to_numpy(dtype=float)
        sp = g["split"].to_numpy()
        for i in range(encoder_len - 1, len(g) - horizon):          # i = origin (last encoder step)
            if not contiguous[i - encoder_len + 1: i + horizon].all():
                continue
            if not (sp[i + 1: i + horizon + 1] == split).all():
                continue
            if not allow_past_context and not (sp[i - encoder_len + 1: i + 1] == split).all():
                continue
            Xs.append(vals[i - encoder_len + 1: i + 1])
            ys.append(tgt[i + 1: i + horizon + 1])
            origins.append(int(t[i]))
            groups.append(gid)
    if not Xs:
        return {"X": np.empty((0, encoder_len, len(feature_cols))), "y": np.empty((0, horizon)),
                "origin_time_idx": np.array([], dtype=int), "group_id": np.array([])}
    return {"X": np.stack(Xs), "y": np.stack(ys), "origin_time_idx": np.array(origins),
            "group_id": np.array(groups)}


def check_no_leakage(windows: dict, df: pd.DataFrame, split: str, horizon: int) -> bool:
    """True when no target index of the windows lies outside `split` or before the split's start."""
    sp_idx = df.loc[df["split"] == split, "time_idx"]
    lo, hi = sp_idx.min(), sp_idx.max()
    first_target = windows["origin_time_idx"] + 1
    last_target = windows["origin_time_idx"] + horizon
    return bool((first_target >= lo).all() and (last_target <= hi).all())
