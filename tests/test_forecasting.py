import copy

import numpy as np
import pandas as pd
import pytest

from src.forecasting.evaluate import baseline_forecasts


def test_baseline_forecast_shapes_and_alignment(cfgs, processed):
    df, _ = processed
    f = baseline_forecasts(df, cfgs["forecasting"], "test")
    n, H = f["y"].shape
    assert H == 12 and n > 0
    for name, arr in f["forecasts"].items():
        assert arr.shape == (n, H) and np.isfinite(arr).all(), name
    assert np.allclose(f["forecasts"]["persistence"][:, 0], f["forecasts"]["persistence"][:, -1])


def test_persistence_uses_last_observed_value(cfgs, processed):
    df, _ = processed
    from src.data.build_windows import make_windows
    w = make_windows(df, "request_rate", 288 + 12, 12, "test")
    f = baseline_forecasts(df, cfgs["forecasting"], "test")
    assert np.allclose(f["forecasts"]["persistence"][:, 0], w["X"][:, -1, 0])


@pytest.mark.slow
def test_tft_training_and_quantile_prediction_are_valid(cfgs, processed, tmp_path):
    from src.forecasting.predict import load_model, predict_split
    from src.forecasting.train_tft import train
    df, meta = processed
    c = copy.deepcopy(cfgs)
    c["forecasting"]["training"]["limit_train_batches"] = 4
    info = train(c, df, meta, max_epochs=1, model_dir=tmp_path, quiet=True)
    assert (tmp_path / "tft_best.ckpt").exists() and (tmp_path / "dataset_parameters.pt").exists()
    assert info["n_train_windows"] > 0
    model, params = load_model(c, tmp_path)
    pred = predict_split(model, params, df, meta["boundaries"], c["forecasting"], "test")
    H = c["forecasting"]["horizon"]
    assert set(pred["horizon_step"].unique()) == set(range(1, H + 1))
    assert np.isfinite(pred[["lower", "median", "upper"]].to_numpy()).all()
    assert (pred["lower"] <= pred["median"] + 1e-6).all() and (pred["median"] <= pred["upper"] + 1e-6).all()
    # every target lies in the test period and timestamps advance by one interval per horizon step
    test_start = meta["boundaries"]["test_start_idx"]
    assert (pred["target_time_idx"] >= test_start).all()
    one = pred[(pred.group_id == pred.group_id.iloc[0]) & (pred.origin_time_idx == pred.origin_time_idx.iloc[0])].sort_values("horizon_step")
    assert (one["timestamp"].diff().dropna() == pd.Timedelta(minutes=5)).all()
    # actuals attached to the predictions are the true observations
    chk = pred.merge(df[["group_id", "time_idx", "request_rate"]], left_on=["group_id", "target_time_idx"], right_on=["group_id", "time_idx"])
    assert np.allclose(chk["actual"], chk["request_rate"], rtol=1e-6)
