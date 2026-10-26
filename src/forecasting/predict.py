"""TFT inference: quantile forecasts for validation and test periods (Part B).

Output (long format, one row per origin x horizon step):
  group_id, service_id, split, origin_time_idx, origin_time, horizon_step, timestamp (target time),
  actual, lower, median, upper
Quantile crossing is removed by sorting the three quantiles per forecast.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from ..common import load_all_configs, resolve
from .train_tft import _import_pf, build_eval_dataset, prepare_frame


def load_model(cfgs: dict, model_dir: str | Path | None = None):
    TFT, TimeSeriesDataSet, _, _ = _import_pf()
    d = resolve(model_dir or cfgs["forecasting"]["paths"]["model_dir"])
    model = TFT.load_from_checkpoint(str(d / "tft_best.ckpt"), map_location="cpu")
    model.eval()
    params = torch.load(d / "dataset_parameters.pt", weights_only=False)
    return model, params


def predict_split(model, params, df: pd.DataFrame, bounds: dict, fcfg: dict, split: str) -> pd.DataFrame:
    _, TimeSeriesDataSet, _, _ = _import_pf()
    df = prepare_frame(df)
    start_key = {"validation": "validation_start_idx", "test": "test_start_idx"}[split]
    end_key = {"validation": "validation_end_idx", "test": "test_end_idx"}[split]
    upto = df[(df["time_idx"] <= bounds[end_key])]
    ds = TimeSeriesDataSet.from_parameters(params, upto, min_prediction_idx=bounds[start_key],
                                           stop_randomization=True, predict=False)
    dl = ds.to_dataloader(train=False, batch_size=512, num_workers=0)
    res = model.predict(dl, mode="quantiles", return_index=True, trainer_kwargs=dict(
        accelerator="cpu", logger=False, enable_progress_bar=False, enable_model_summary=False))
    q = res.output.detach().cpu().numpy()                      # (N, H, 3) original scale
    q = np.sort(q, axis=-1)
    idx = res.index.reset_index(drop=True)                     # time_idx of first decoder step + group
    H = int(fcfg["horizon"])
    interval = pd.Timedelta(minutes=5)
    lookup = df.set_index(["group_id", "time_idx"])
    rows = []
    gid = idx["group_id"].astype(str).to_numpy()
    first = idx["time_idx"].to_numpy()
    # fast lookup of actuals / timestamps per group
    by_g = {g: sub.set_index("time_idx") for g, sub in df.groupby("group_id")}
    for n in range(len(idx)):
        g = by_g[gid[n]]
        for h in range(H):
            ti = int(first[n]) + h
            if ti not in g.index:
                continue
            rows.append((gid[n], g["service_id"].iloc[0], split, int(first[n]) - 1, h + 1, ti,
                         g.at[ti, "timestamp"], g.at[ti, fcfg["target"]],
                         q[n, h, 0], q[n, h, 1], q[n, h, 2]))
    out = pd.DataFrame(rows, columns=["group_id", "service_id", "split", "origin_time_idx", "horizon_step",
                                      "target_time_idx", "timestamp", "actual", "lower", "median", "upper"])
    tmap = df.drop_duplicates("time_idx").set_index("time_idx")["timestamp"]
    out["origin_time"] = out["origin_time_idx"].map(tmap)
    return out


def predict_and_save(cfgs: dict, df: pd.DataFrame, meta: dict, model_dir=None) -> pd.DataFrame:
    model, params = load_model(cfgs, model_dir)
    parts = [predict_split(model, params, df, meta["boundaries"], cfgs["forecasting"], s)
             for s in ("validation", "test")]
    out = pd.concat(parts, ignore_index=True)
    path = resolve(cfgs["forecasting"]["paths"]["predictions_file"])
    path.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(path, index=False)
    test = out[out["split"] == "test"]
    (resolve("results/metrics")).mkdir(parents=True, exist_ok=True)
    test.drop(columns=["origin_time_idx"]).to_csv(resolve("results/metrics/tft_test_predictions.csv"), index=False)
    print(f"[tft] wrote {len(out)} forecast rows -> {path}")
    return out


def main() -> None:
    cfgs = load_all_configs()
    from ..data.load_data import load_processed
    df = load_processed(cfgs["data"])
    meta = json.load(open(resolve("data/processed/preprocess_config.json")))
    predict_and_save(cfgs, df, meta)


if __name__ == "__main__":
    main()
