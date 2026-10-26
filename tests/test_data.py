import numpy as np
import pandas as pd
import pytest

from src.data.build_windows import assert_chronological_split, check_no_leakage, make_windows
from src.data.preprocess import preprocess
from src.data.validate_data import validate_dataframe


def _base(n=50):
    ts = pd.date_range("2026-01-05", periods=n, freq="5min")
    return pd.DataFrame({"timestamp": ts, "service_id": "a", "request_rate": np.linspace(100, 200, n),
                         "cpu_utilization": 0.5, "memory_utilization": 0.4, "latency_ms": 50.0})


def test_validation_reports_missing_required_field(cfgs):
    rep = validate_dataframe(_base().drop(columns=["request_rate"]), cfgs["data"])
    assert not rep.ok and any("request_rate" in e for e in rep.errors)


def test_validation_reports_invalid_timestamps(cfgs):
    df = _base().astype({"timestamp": object})
    df.loc[3, "timestamp"] = "not-a-date"
    rep = validate_dataframe(df, cfgs["data"])
    assert not rep.ok and rep.stats["n_bad_timestamps"] == 1


def test_validation_reports_duplicates_gaps_units_and_negatives(cfgs):
    df = _base()
    df = pd.concat([df, df.iloc[[5]]]).drop(df.index[20:23])
    df.loc[0, "cpu_utilization"] = 55.0           # percent instead of fraction
    df.loc[1, "latency_ms"] = -4.0
    rep = validate_dataframe(df, cfgs["data"])
    assert rep.stats["n_duplicate_timestamps"] == 1
    assert rep.stats["n_gaps"] >= 1
    assert any("outside [0, 1]" in e for e in rep.errors)
    assert any("negative" in e for e in rep.errors)
    with pytest.raises(ValueError):
        rep.raise_if_errors()


def test_chronological_split_has_no_overlap(processed):
    df, meta = processed
    assert_chronological_split(df)
    b = meta["boundaries"]
    assert df[df.split == "train"].time_idx.max() == b["train_end_idx"]
    assert df[df.split == "test"].time_idx.min() == b["test_start_idx"]
    frac = df.groupby("split").size() / len(df)
    assert abs(frac["train"] - 0.70) < 0.01 and abs(frac["validation"] - 0.15) < 0.01


def test_training_windows_never_contain_future_observations(processed):
    df, _ = processed
    tr = make_windows(df, "request_rate", 48, 12, "train", allow_past_context=False)
    assert len(tr["X"]) > 0 and check_no_leakage(tr, df, "train", 12)
    test_start = df[df.split == "test"].time_idx.min()
    assert (tr["origin_time_idx"] + 12 < test_start).all()
    te = make_windows(df, "request_rate", 48, 12, "test")
    assert check_no_leakage(te, df, "test", 12)
    assert te["origin_time_idx"].min() >= test_start - 1       # targets all inside test


def test_scalers_fit_on_training_partition_only(processed):
    df, meta = processed
    m_train = df.loc[df.split == "train", "request_rate"].mean()
    m_all = df["request_rate"].mean()
    assert meta["scalers_train_only"]["request_rate"]["mean"] == pytest.approx(m_train)
    assert abs(m_train - m_all) > 1e-6


def test_missing_telemetry_is_not_zero_filled(cfgs):
    df = _base(200)
    df = df.drop(df.index[50:52])                     # 2-step gap -> interpolated
    df.loc[df.index[100], "cpu_utilization"] = np.nan  # isolated NaN
    out, meta = preprocess(df, cfgs["data"])
    assert (out["cpu_utilization"] > 0).all() and (out["request_rate"] > 0).all()
    assert meta["rows_imputed"] >= 3 and out["imputed"].sum() == meta["rows_imputed"]


def test_long_gap_starts_new_segment_and_windows_do_not_cross_it(cfgs):
    df = _base(300)
    df = df.drop(df.index[100:120])                   # 20 steps > max_gap_steps
    out, meta = preprocess(df, cfgs["data"])
    assert meta["rows_dropped_long_gaps"] == 20 and out["group_id"].nunique() == 2
    w = make_windows(out, "request_rate", 20, 5, "train", allow_past_context=True)
    for gid, o in zip(w["group_id"], w["origin_time_idx"]):
        seg = out[out.group_id == gid].time_idx
        assert seg.min() <= o - 19 and o + 5 <= seg.max()


def test_preprocess_config_is_saved_and_complete(processed):
    _, meta = processed
    for k in ("boundaries", "scalers_train_only", "split", "cleaning", "price_model", "data_config_hash"):
        assert k in meta


def test_cost_derived_from_explicit_price_model(cfgs):
    df = _base(30)
    df["replica_count"], df["pending_replicas"] = 10, 2
    out, meta = preprocess(df, cfgs["data"])
    assert meta["cost_derived_from_price_model"]
    assert out["resource_cost"].iloc[0] == pytest.approx(12 * 0.05 * 5 / 60)
