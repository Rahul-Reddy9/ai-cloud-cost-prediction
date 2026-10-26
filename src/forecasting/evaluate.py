"""Forecast evaluation (Experiment 1): TFT vs persistence / moving average / seasonal-naive.

All methods are scored on the SAME test targets (origins whose 12 targets lie in the test split).
TFT metrics additionally include quantile loss, 80% prediction-interval coverage and interval width.
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from ..common import load_all_configs, resolve, save_json
from ..data.build_windows import make_windows
from ..evaluation.metrics import (interval_coverage, interval_width, mae, mean_quantile_loss, pinball_loss,
                                  rmse)


def baseline_forecasts(df: pd.DataFrame, fcfg: dict, split: str = "test") -> dict:
    """Return dict name -> (N,H) forecast array aligned with the windows (and y, origin ids)."""
    H, enc = int(fcfg["horizon"]), int(fcfg["encoder_length"])
    ma_w = int(fcfg["baselines"]["moving_average_window"])
    season = int(fcfg["baselines"]["seasonal_period_steps"])
    tgt = fcfg["target"]
    # encoder must cover the seasonal lag so seasonal-naive can read "same time yesterday"
    win = make_windows(df, tgt, max(enc, season + H), H, split, [tgt], allow_past_context=True)
    X, y = win["X"][:, :, 0], win["y"]
    out = {
        "persistence": np.repeat(X[:, -1:], H, axis=1),
        "moving_average": np.repeat(X[:, -ma_w:].mean(axis=1, keepdims=True), H, axis=1),
    }
    # seasonal naive: value at (target time - season); the encoder holds the last (season+H) steps
    idx = np.arange(1, H + 1)[None, :] - season + (X.shape[1] - 1)
    out["seasonal_naive"] = np.take_along_axis(X, np.repeat(idx, len(X), axis=0), axis=1)
    return {"forecasts": out, "y": y, "origin_time_idx": win["origin_time_idx"], "group_id": win["group_id"]}


def evaluate_forecasts(cfgs: dict | None = None, split: str = "test") -> dict:
    cfgs = cfgs or load_all_configs()
    fcfg = cfgs["forecasting"]
    from ..data.load_data import load_processed
    df = load_processed(cfgs["data"])
    pred = pd.read_parquet(resolve(fcfg["paths"]["predictions_file"]))
    pred = pred[pred["split"] == split]
    H = int(fcfg["horizon"])
    qs = list(fcfg["quantiles"])
    base = baseline_forecasts(df, fcfg, split)

    # Align baselines and TFT on identical (group, origin) pairs
    key_b = pd.MultiIndex.from_arrays([base["group_id"], base["origin_time_idx"]])
    key_t = pd.MultiIndex.from_frame(pred[["group_id", "origin_time_idx"]].drop_duplicates())
    common = key_b.intersection(key_t)
    pos_b = key_b.get_indexer(common)
    p = pred.set_index(["group_id", "origin_time_idx", "horizon_step"]).sort_index()
    cg = common.get_level_values(0)
    co = common.get_level_values(1)
    mk = lambda col: np.stack([p.loc[(g, o), col].to_numpy() for g, o in zip(cg, co)])
    tft_lo, tft_med, tft_hi, y_t = mk("lower"), mk("median"), mk("upper"), mk("actual")
    y = base["y"][pos_b]
    assert np.allclose(y, y_t, rtol=1e-4, atol=1e-3), "TFT and baseline targets are misaligned"

    rows = {}
    methods = {"tft_median": tft_med}
    for name, f in base["forecasts"].items():
        methods[name] = f[pos_b]
    for name, f in methods.items():
        rows[name] = {"mae": mae(y, f), "rmse": rmse(y, f),
                      "mae_by_horizon": [mae(y[:, h], f[:, h]) for h in range(H)],
                      "rmse_by_horizon": [rmse(y[:, h], f[:, h]) for h in range(H)]}
    quant = {qs[0]: tft_lo, qs[1]: tft_med, qs[2]: tft_hi}
    rows["tft_median"].update({
        "quantile_loss_mean": mean_quantile_loss(y, quant),
        "pinball": {str(q): pinball_loss(y, v, q) for q, v in quant.items()},
        "interval_nominal_coverage": qs[2] - qs[0],
        "interval_coverage": interval_coverage(y, tft_lo, tft_hi),
        "interval_mean_width": interval_width(tft_lo, tft_hi),
        "coverage_by_horizon": [interval_coverage(y[:, h], tft_lo[:, h], tft_hi[:, h]) for h in range(H)],
    })
    best_base = min(("persistence", "moving_average", "seasonal_naive"), key=lambda k: rows[k]["mae"])
    result = {"split": split, "n_windows": int(len(y)), "horizon": H, "methods": rows,
              "best_baseline_by_mae": best_base,
              "tft_vs_best_baseline_mae_improvement_pct":
                  100 * (rows[best_base]["mae"] - rows["tft_median"]["mae"]) / rows[best_base]["mae"],
              "tft_vs_persistence_mae_improvement_pct":
                  100 * (rows["persistence"]["mae"] - rows["tft_median"]["mae"]) / rows["persistence"]["mae"]}
    save_json(result, "results/metrics/forecast_eval.json")
    return result


def main() -> None:
    r = evaluate_forecasts()
    print(f"[forecast eval] windows={r['n_windows']}")
    for k, v in r["methods"].items():
        print(f"  {k:15s} MAE={v['mae']:.2f} RMSE={v['rmse']:.2f}")
    t = r["methods"]["tft_median"]
    print(f"  TFT quantile loss={t['quantile_loss_mean']:.2f} coverage={t['interval_coverage']:.3f} "
          f"(nominal {t['interval_nominal_coverage']:.2f}) width={t['interval_mean_width']:.1f}")


if __name__ == "__main__":
    main()
