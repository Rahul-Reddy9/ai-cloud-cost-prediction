"""Dataset validation (Part A): timestamps, units, duplicates, missing observations."""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd


@dataclass
class ValidationReport:
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    stats: dict = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return not self.errors

    def raise_if_errors(self) -> None:
        if self.errors:
            raise ValueError("Dataset validation failed:\n  - " + "\n  - ".join(self.errors))

    def to_dict(self) -> dict:
        return {"ok": self.ok, "errors": self.errors, "warnings": self.warnings, "stats": self.stats}


def validate_dataframe(df: pd.DataFrame, cfg: dict) -> ValidationReport:
    """Validate raw data against the internal schema. Errors block processing; warnings are documented."""
    rep = ValidationReport()
    schema = cfg["schema"]
    interval = pd.Timedelta(minutes=int(cfg["dataset"]["interval_minutes"]))

    missing_req = [c for c in schema["required"] if c not in df.columns]
    if missing_req:
        rep.errors.append(f"missing required columns: {missing_req}")
        return rep
    missing_opt = [c for c in schema["optional"] if c not in df.columns]
    if missing_opt:
        rep.warnings.append(f"optional columns absent (document and derive/skip): {missing_opt}")
    rep.stats["n_rows"] = int(len(df))

    ts = pd.to_datetime(df["timestamp"], errors="coerce")
    n_bad_ts = int(ts.isna().sum())
    if n_bad_ts:
        rep.errors.append(f"{n_bad_ts} unparsable/missing timestamps")
    rep.stats["n_bad_timestamps"] = n_bad_ts

    svc = df["service_id"] if "service_id" in df.columns else pd.Series("svc1", index=df.index)
    work = pd.DataFrame({"service_id": svc, "timestamp": ts}).dropna()

    n_dup = int(work.duplicated(["service_id", "timestamp"]).sum())
    rep.stats["n_duplicate_timestamps"] = n_dup
    if n_dup:
        rep.warnings.append(f"{n_dup} duplicate (service, timestamp) rows (last kept during preprocessing)")

    n_nonmono, n_gap, max_gap, n_irregular = 0, 0, pd.Timedelta(0), 0
    for _, g in work.groupby("service_id"):
        t = g["timestamp"]
        n_nonmono += int((t.diff().dropna() < pd.Timedelta(0)).sum())
        d = t.sort_values().drop_duplicates().diff().dropna()
        n_gap += int((d > interval).sum())
        n_irregular += int(((d != interval) & (d > pd.Timedelta(0))).sum())
        if len(d):
            max_gap = max(max_gap, d.max())
    rep.stats.update({"n_non_monotonic": n_nonmono, "n_gaps": n_gap, "n_irregular_steps": n_irregular,
                      "max_gap_minutes": float(max_gap / pd.Timedelta(minutes=1))})
    if n_nonmono:
        rep.warnings.append(f"{n_nonmono} out-of-order timestamps (data will be sorted)")
    if n_gap:
        rep.warnings.append(f"{n_gap} gaps larger than the {interval} sampling interval "
                            f"(longest {max_gap}); short gaps are interpolated, long gaps split segments")

    nan_counts = {c: int(df[c].isna().sum()) for c in schema["required"] + schema["optional"]
                  if c in df.columns and c != "timestamp"}
    rep.stats["nan_counts"] = nan_counts
    for c, n in nan_counts.items():
        if n:
            rep.warnings.append(f"{n} missing values in '{c}' (never treated as zero)")

    for c in schema["bounded_0_1"]:
        if c in df.columns:
            v = df[c].dropna()
            bad = int(((v < -1e-6) | (v > 1 + 1e-6)).sum())
            if bad:
                rep.errors.append(f"{bad} values of '{c}' outside [0, 1] (check units: percent vs fraction)")
    for c in schema["non_negative"]:
        if c in df.columns:
            bad = int((df[c].dropna() < 0).sum())
            if bad:
                rep.errors.append(f"{bad} negative values in '{c}'")
    if "anomaly_label" in df.columns:
        vals = set(pd.unique(df["anomaly_label"].dropna()))
        if not vals <= {0, 1}:
            rep.errors.append(f"anomaly_label must be 0/1, found {sorted(vals)[:5]}")
    return rep
