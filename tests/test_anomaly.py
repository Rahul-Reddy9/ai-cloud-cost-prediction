import copy

import numpy as np
import pandas as pd
import pytest

from src.anomaly.evaluate_detector import inject_synthetic_anomalies
from src.anomaly.score_anomalies import score_frame
from src.anomaly.train_detector import train_detectors
from src.evaluation.metrics import detection_metrics


def naive_predictions(df: pd.DataFrame, noise=0.05) -> pd.DataFrame:
    """Stand-in 1-step forecasts (previous value) in the same long format TFT produces."""
    rows = []
    for gid, g in df.sort_values(["group_id", "time_idx"]).groupby("group_id"):
        g = g[g.split.isin(["validation", "test"])]
        med = g["request_rate"].shift(1).bfill()
        rows.append(pd.DataFrame({"group_id": gid, "service_id": g["service_id"], "split": g["split"], "origin_time_idx": g["time_idx"] - 1,
                                  "horizon_step": 1, "target_time_idx": g["time_idx"], "timestamp": g["timestamp"],
                                  "actual": g["request_rate"], "lower": med * (1 - noise), "median": med, "upper": med * (1 + noise)}))
    return pd.concat(rows, ignore_index=True)


@pytest.fixture(scope="module")
def trained(cfgs, processed, tmp_path_factory):
    df, meta = processed
    c = copy.deepcopy(cfgs)
    c["anomaly"]["paths"]["model_dir"] = str(tmp_path_factory.mktemp("anom"))
    pred = naive_predictions(df)
    info = train_detectors(c, df, pred, meta)
    return c, df, meta, pred, info


def test_detector_is_not_fitted_on_test_or_anomalous_rows(trained):
    c, df, meta, pred, info = trained
    n_normal_train = int(((df.split == "train") & (df.anomaly_label == 0)).sum())
    assert info["raw"]["n_fit"] == n_normal_train
    assert info["residual"]["fit_until_time_idx"] < meta["boundaries"]["test_start_idx"]
    assert info["labels_are_synthetic"] is True


def test_scores_are_finite_and_correctly_shaped(trained):
    c, df, meta, pred, _ = trained
    sc = score_frame(c, df, pred, meta)
    assert len(sc) == len(df)
    assert np.isfinite(sc["score_raw"]).all() and np.isfinite(sc["z_raw"]).all()
    assert (sc["z_raw"].between(0, 2)).all()
    r = sc["score_resid"].dropna()
    assert len(r) > 0 and np.isfinite(r).all() and set(sc["flag_raw"].unique()) <= {0, 1}
    assert sc.loc[sc.split == "train", "score_resid"].isna().all()       # no forecasts => no residual score


def test_detector_handles_single_row_input(trained):
    c, df, meta, pred, _ = trained
    sc = score_frame(c, df.iloc[:1].copy(), None, meta)
    assert sc.shape[0] == 1 and np.isfinite(sc["score_raw"]).all()


def test_scores_separate_injected_anomalies_better_than_chance(trained):
    from sklearn.metrics import roc_auc_score
    c, df, meta, pred, _ = trained
    sc = score_frame(c, df, pred, meta)
    t = sc[(sc.split == "test") & sc.score_resid.notna()]
    assert roc_auc_score(t.anomaly_label, t.score_raw) > 0.6
    assert roc_auc_score(t.anomaly_label, t.score_resid) > 0.55


def test_controlled_injection_for_unlabelled_data(processed):
    df, _ = processed
    d = inject_synthetic_anomalies(df.drop(columns=["anomaly_label"]), seed=1)
    assert d["anomaly_label"].sum() > 0 and set(d["anomaly_label"].unique()) <= {0, 1}
