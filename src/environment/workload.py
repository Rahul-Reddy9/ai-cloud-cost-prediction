"""Workload generation (Part D) and trace containers used by the simulator/gym environment.

`generate_dataset` builds a SYNTHETIC multi-service workload with known, injected anomalies. It is a
stand-in for a real trace (Azure / Alibaba / Google); replace it by setting `dataset.source: csv`
in configs/data.yaml. Synthetic data must be labelled as such in any report.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd

from .cost_model import PriceModel
from .simulator import ClusterSimulator, SimulatorConfig

START_TIMESTAMP = "2026-01-05 00:00:00"   # a Monday


# ----------------------------------------------------------------------------------------------
# Synthetic demand
# ----------------------------------------------------------------------------------------------
def _ar1(n: int, rho: float, sigma: float, rng: np.random.Generator) -> np.ndarray:
    e = np.zeros(n)
    innov = rng.normal(0.0, sigma * np.sqrt(1 - rho ** 2), n)
    for i in range(1, n):
        e[i] = rho * e[i - 1] + innov[i]
    return e


def generate_service_demand(n_days: int, interval_minutes: int, scale: float, phase_hours: float,
                            seed: int, p: dict) -> pd.DataFrame:
    """Return a frame with timestamp, request_rate and injected anomaly info for ONE service."""
    rng = np.random.default_rng(seed)
    per_day = int(24 * 60 / interval_minutes)
    n = n_days * per_day
    ts = pd.date_range(START_TIMESTAMP, periods=n, freq=f"{interval_minutes}min")
    hours = (np.arange(n) * interval_minutes / 60.0)
    hod = hours % 24
    daily = (1 + p["daily_amplitude"] * np.sin(2 * np.pi * (hod - 9 + phase_hours) / 24)
             + p["second_harmonic"] * np.sin(4 * np.pi * (hod - 6 + phase_hours) / 24))
    day_idx = (np.arange(n) // per_day)
    weekend = np.where(day_idx % 7 >= 5, p["weekend_factor"], 1.0)
    trend = 1 + p["trend_per_30_days"] * (day_idx / 30.0)
    noise = np.exp(_ar1(n, p["noise_ar"], p["noise_sigma"], rng))
    base = p["base_rate_rps"] * scale * daily * weekend * trend * noise

    # normal traffic events (promotions): part of the legitimate workload, not labelled anomalies
    events = np.ones(n)
    n_events = rng.poisson(p["events_per_week"] * n_days / 7)
    for _ in range(n_events):
        dur = int(rng.integers(*p["event_duration_steps"]))
        start = int(rng.integers(0, max(1, n - dur)))
        mag = rng.uniform(*p["event_magnitude"])
        ramp = np.sin(np.linspace(0, np.pi, dur)) ** 0.5
        events[start:start + dur] *= 1 + (mag - 1) * ramp

    demand_mult = np.ones(n)
    latency_mult = np.ones(n)
    cpu_add = np.zeros(n)
    label = np.zeros(n, dtype=int)
    a_type = np.array([""] * n, dtype=object)
    a = p["anomalies"]
    for _ in range(rng.poisson(a["per_day"] * n_days)):
        kind = str(rng.choice(a["types"]))
        dur = int(rng.integers(*a["duration_steps"]))
        start = int(rng.integers(per_day // 4, max(per_day // 4 + 1, n - dur)))
        sl = slice(start, start + dur)
        if kind == "surge":
            demand_mult[sl] *= rng.uniform(*a["surge_magnitude"])
        elif kind == "drop":
            demand_mult[sl] *= rng.uniform(*a["drop_magnitude"])
        elif kind == "latency_spike":
            latency_mult[sl] *= rng.uniform(*a["latency_multiplier"])
        elif kind == "cpu_saturation":
            cpu_add[sl] += rng.uniform(*a["cpu_additive"])
        label[sl] = 1
        a_type[sl] = kind
    demand = np.maximum(base * events * demand_mult, 1.0)
    return pd.DataFrame({"timestamp": ts, "request_rate": demand, "anomaly_label": label,
                         "anomaly_type": a_type, "_latency_mult": latency_mult, "_cpu_add": cpu_add})


def logged_telemetry(frame: pd.DataFrame, sim_cfg: SimulatorConfig, price: PriceModel,
                     logging_policy: dict, seed: int) -> pd.DataFrame:
    """Replay demand through the simulator under a simple threshold rule to obtain the full schema
    (cpu, memory, latency, replicas, pending, cost) as if an operator had been running the cluster."""
    sim = ClusterSimulator(sim_cfg, price, seed=seed)
    sim.reset(sim_cfg.initial_replicas)
    target = logging_policy["target_util"]
    max_step = logging_policy["max_step"]
    cpu_prev = 0.6
    rows = []
    for d, lm, ca in zip(frame["request_rate"].to_numpy(), frame["_latency_mult"].to_numpy(),
                         frame["_cpu_add"].to_numpy()):
        desired = int(np.ceil(sim.total_replicas * max(cpu_prev, 1e-3) / target))
        delta = int(np.clip(desired - sim.total_replicas, -1, max_step))
        sim.request_scaling(delta)
        rec = sim.step(d)
        cpu_prev = rec["cpu_utilization"]
        rows.append((np.clip(rec["cpu_utilization"] + ca, 0, 1), rec["memory_utilization"],
                     rec["latency_ms"] * lm, rec["ready"], rec["pending"], rec["cost"]))
    out = frame.copy()
    arr = np.array(rows)
    out["cpu_utilization"], out["memory_utilization"], out["latency_ms"] = arr[:, 0], arr[:, 1], arr[:, 2]
    out["replica_count"] = arr[:, 3].astype(int)
    out["pending_replicas"] = arr[:, 4].astype(int)
    out["resource_cost"] = arr[:, 5]
    return out


def generate_dataset(data_cfg: dict, sim_cfg_dict: dict) -> pd.DataFrame:
    """Generate the multi-service SYNTHETIC dataset with the full internal schema."""
    s = data_cfg["synthetic"]
    interval = int(data_cfg["dataset"]["interval_minutes"])
    sim_cfg = SimulatorConfig.from_config(sim_cfg_dict)
    pm = data_cfg["price_model"]
    price = PriceModel(float(pm["price_per_replica_hour"]), float(pm["billing_interval_minutes"]),
                       pm["currency"])
    frames = []
    for i in range(int(s["n_services"])):
        sid = f"svc{i + 1}"
        f = generate_service_demand(s["n_days"], interval, s["service_scales"][i % len(s["service_scales"])],
                                    s["service_phase_hours"][i % len(s["service_phase_hours"])],
                                    seed=int(s["seed"]) * 100 + i, p=s)
        f = logged_telemetry(f, sim_cfg, price, s["logging_policy"], seed=int(s["seed"]) * 100 + i)
        f.insert(1, "service_id", sid)
        f = f.drop(columns=["_latency_mult", "_cpu_add"])
        frames.append(f)
    df = pd.concat(frames, ignore_index=True)
    rng = np.random.default_rng(int(s["seed"]) + 999)
    # telemetry gaps: missing timestamps and isolated NaN values (never silently zero)
    drop = rng.random(len(df)) < float(s["missing_timestamp_fraction"])
    df = df.loc[~drop].reset_index(drop=True)
    nanmask = rng.random(len(df)) < float(s["nan_value_fraction"])
    df.loc[nanmask, "cpu_utilization"] = np.nan
    return df


# ----------------------------------------------------------------------------------------------
# Trace container consumed by the gym environment / baselines
# ----------------------------------------------------------------------------------------------
@dataclass
class WorkloadTrace:
    """One contiguous workload segment, with decision-time-safe side information.

    forecasts[i] is the TFT forecast issued at time index i (using data up to and including i) for
    steps i+1..i+H; shape (T, H, 3) with quantile order (low, median, high); NaN when unavailable.
    z_raw[i] / z_resid[i] are anomaly scores computable at time i.
    """
    name: str
    timestamps: np.ndarray
    demand: np.ndarray
    hour_sin: np.ndarray
    hour_cos: np.ndarray
    forecasts: Optional[np.ndarray] = None
    z_raw: Optional[np.ndarray] = None
    z_resid: Optional[np.ndarray] = None
    anomaly_label: Optional[np.ndarray] = None

    def __len__(self) -> int:
        return len(self.demand)

    @property
    def horizon(self) -> int:
        return 0 if self.forecasts is None else self.forecasts.shape[1]

    def last_valid_origin(self) -> int:
        """Largest origin index i such that the full H-step forecast exists and targets are in-trace."""
        if self.forecasts is None:
            return len(self) - 2
        ok = np.where(~np.isnan(self.forecasts[:, :, 1]).any(axis=1))[0]
        return int(min(ok.max(), len(self) - 2)) if len(ok) else -1

    def first_valid_origin(self) -> int:
        if self.forecasts is None:
            return 0
        ok = np.where(~np.isnan(self.forecasts[:, :, 1]).any(axis=1))[0]
        return int(ok.min()) if len(ok) else 0

    def slice(self, start: int, stop: int, name: str | None = None) -> "WorkloadTrace":
        def s(a):
            return None if a is None else a[start:stop]
        return WorkloadTrace(name or f"{self.name}[{start}:{stop}]", self.timestamps[start:stop],
                             self.demand[start:stop], self.hour_sin[start:stop], self.hour_cos[start:stop],
                             s(self.forecasts), s(self.z_raw), s(self.z_resid), s(self.anomaly_label))


def make_plain_trace(name: str, demand: np.ndarray, interval_minutes: int = 5,
                     start: str = START_TIMESTAMP, forecast_fn=None, horizon: int = 12) -> WorkloadTrace:
    """Helper for tests/demos: build a trace from a demand array (optionally with synthetic forecasts)."""
    n = len(demand)
    ts = pd.date_range(start, periods=n, freq=f"{interval_minutes}min").to_numpy()
    hod = (np.arange(n) * interval_minutes / 60.0) % 24
    fc = None
    if forecast_fn is not None:
        fc = np.full((n, horizon, 3), np.nan)
        for i in range(n - horizon):
            fc[i] = forecast_fn(i, demand, horizon)
    return WorkloadTrace(name, ts, np.asarray(demand, float), np.sin(2 * np.pi * hod / 24),
                         np.cos(2 * np.pi * hod / 24), fc)


def traces_from_data(df: pd.DataFrame, predictions: pd.DataFrame | None, scores: pd.DataFrame | None,
                     split: str, horizon: int = 12) -> list[WorkloadTrace]:
    """Build one WorkloadTrace per (service, segment) for `split`, attaching TFT forecasts and anomaly scores.

    Forecast row i (origin = time index of row i) holds quantiles (low, median, high) for steps 1..H.
    """
    traces = []
    d = df[df["split"] == split].sort_values(["group_id", "time_idx"])
    pred = None
    if predictions is not None:
        pred = predictions[predictions["split"] == split]
    for gid, g in d.groupby("group_id"):
        g = g.reset_index(drop=True)
        n = len(g)
        t0 = int(g["time_idx"].iloc[0])
        fc = None
        if pred is not None:
            fc = np.full((n, horizon, 3), np.nan)
            pg = pred[pred["group_id"] == gid]
            pos = (pg["origin_time_idx"].to_numpy() - t0).astype(int)
            hs = pg["horizon_step"].to_numpy().astype(int) - 1
            ok = (pos >= 0) & (pos < n)
            fc[pos[ok], hs[ok], 0] = pg["lower"].to_numpy()[ok]
            fc[pos[ok], hs[ok], 1] = pg["median"].to_numpy()[ok]
            fc[pos[ok], hs[ok], 2] = pg["upper"].to_numpy()[ok]
        zr = zs = None
        if scores is not None:
            sg = scores[scores["group_id"] == gid].set_index("time_idx").reindex(g["time_idx"])
            zr, zs = sg["z_raw"].to_numpy(), sg["z_resid"].to_numpy()
        traces.append(WorkloadTrace(
            name=f"{gid}:{split}", timestamps=g["timestamp"].to_numpy(), demand=g["request_rate"].to_numpy(float),
            hour_sin=g["hour_sin"].to_numpy(float), hour_cos=g["hour_cos"].to_numpy(float),
            forecasts=fc, z_raw=zr, z_resid=zs,
            anomaly_label=g["anomaly_label"].to_numpy(int) if "anomaly_label" in g else None))
    return traces
