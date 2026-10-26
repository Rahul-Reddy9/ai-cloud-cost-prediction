"""TFT training (Part B).

Usage:  python -m src.forecasting.train_tft [--epochs N] [--config-dir configs]

Builds historical windows from the TRAINING partition, tunes/early-stops on the validation partition,
saves the checkpoint + dataset parameters + metadata, and finally writes quantile forecasts for the
validation and test periods (see predict.py).
"""
from __future__ import annotations

import argparse
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from ..common import environment_info, load_all_configs, resolve, save_json, set_global_seed

warnings.filterwarnings("ignore", message=".*does not have many workers.*")
warnings.filterwarnings("ignore", category=UserWarning)


def _import_pf():
    from pytorch_forecasting import TemporalFusionTransformer, TimeSeriesDataSet
    from pytorch_forecasting.data import GroupNormalizer
    from pytorch_forecasting.metrics import QuantileLoss
    return TemporalFusionTransformer, TimeSeriesDataSet, GroupNormalizer, QuantileLoss


def prepare_frame(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["service_id"] = df["service_id"].astype(str)
    df["group_id"] = df["group_id"].astype(str)
    return df.sort_values(["group_id", "time_idx"]).reset_index(drop=True)


def build_training_dataset(df: pd.DataFrame, fcfg: dict):
    _, TimeSeriesDataSet, GroupNormalizer, _ = _import_pf()
    train_df = df[df["split"] == "train"]
    return TimeSeriesDataSet(
        train_df, time_idx="time_idx", target=fcfg["target"], group_ids=["group_id"],
        min_encoder_length=fcfg["encoder_length"], max_encoder_length=fcfg["encoder_length"],
        min_prediction_length=fcfg["horizon"], max_prediction_length=fcfg["horizon"],
        static_categoricals=["service_id"],
        time_varying_known_reals=list(fcfg["known_reals"]),
        time_varying_unknown_reals=list(fcfg["unknown_reals"]),
        target_normalizer=GroupNormalizer(groups=["group_id"], transformation="softplus"),
        add_relative_time_idx=True, add_target_scales=True, add_encoder_length=False,
        allow_missing_timesteps=True,
    )


def build_eval_dataset(training, df: pd.DataFrame, bounds: dict, split: str):
    """Dataset whose prediction targets all lie in `split`; the encoder may read earlier (past) data only."""
    _, TimeSeriesDataSet, _, _ = _import_pf()
    start_key = {"validation": "validation_start_idx", "test": "test_start_idx"}[split]
    end_key = {"validation": "validation_end_idx", "test": "test_end_idx"}.get(split)
    upto = df[df["time_idx"] <= (bounds[end_key] if end_key in bounds else bounds["n_grid"])]
    upto = upto[upto["split"].isin(["train", "validation"] if split == "validation"
                                   else ["train", "validation", "test"])]
    return TimeSeriesDataSet.from_dataset(training, upto, min_prediction_idx=bounds[start_key],
                                          stop_randomization=True, predict=False)


def train(cfgs: dict, df: pd.DataFrame, meta: dict, max_epochs: int | None = None,
          model_dir: str | Path | None = None, quiet: bool = False) -> dict:
    import lightning.pytorch as pl
    from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
    TFT, _, _, QuantileLoss = _import_pf()

    fcfg = cfgs["forecasting"]
    tcfg, mcfg = fcfg["training"], fcfg["model"]
    set_global_seed(int(tcfg["seed"]))
    df = prepare_frame(df)
    bounds = meta["boundaries"]
    model_dir = resolve(model_dir or fcfg["paths"]["model_dir"])
    model_dir.mkdir(parents=True, exist_ok=True)

    training = build_training_dataset(df, fcfg)
    validation = build_eval_dataset(training, df, bounds, "validation")
    train_dl = training.to_dataloader(train=True, batch_size=int(tcfg["batch_size"]),
                                      num_workers=int(tcfg["num_workers"]))
    val_dl = validation.to_dataloader(train=False, batch_size=int(tcfg["batch_size"]) * 2,
                                      num_workers=int(tcfg["num_workers"]))

    model = TFT.from_dataset(
        training, learning_rate=float(mcfg["learning_rate"]), hidden_size=int(mcfg["hidden_size"]),
        attention_head_size=int(mcfg["attention_head_size"]), dropout=float(mcfg["dropout"]),
        hidden_continuous_size=int(mcfg["hidden_continuous_size"]),
        loss=QuantileLoss(quantiles=list(fcfg["quantiles"])),
        reduce_on_plateau_patience=int(mcfg["reduce_on_plateau_patience"]), log_interval=-1,
    )
    ckpt = ModelCheckpoint(dirpath=str(model_dir), filename="tft_best", monitor="val_loss", mode="min",
                           save_top_k=1)
    es = EarlyStopping(monitor="val_loss", patience=int(tcfg["early_stopping_patience"]), mode="min")
    trainer = pl.Trainer(
        max_epochs=int(max_epochs or tcfg["max_epochs"]), accelerator=tcfg["accelerator"], devices=1,
        gradient_clip_val=float(tcfg["gradient_clip_val"]), callbacks=[ckpt, es],
        limit_train_batches=tcfg["limit_train_batches"], enable_progress_bar=not quiet,
        logger=False, enable_model_summary=False, default_root_dir=str(model_dir),
    )
    t0 = time.time()
    trainer.fit(model, train_dataloaders=train_dl, val_dataloaders=val_dl)
    elapsed = time.time() - t0

    best_path = Path(ckpt.best_model_path)
    final = model_dir / "tft_best.ckpt"
    if best_path.exists() and best_path != final:
        best_path.replace(final)
    torch.save(training.get_parameters(), model_dir / "dataset_parameters.pt")
    info = {
        "checkpoint": str(final), "best_val_loss": float(ckpt.best_model_score) if ckpt.best_model_score is not None else None,
        "epochs_run": int(trainer.current_epoch), "train_seconds": elapsed,
        "n_train_windows": len(training), "n_val_windows": len(validation),
        "n_parameters": int(sum(p.numel() for p in model.parameters())),
        "forecasting_config": fcfg, "environment": environment_info(),
        "note": "Scalers/normalisers are fitted on the training partition only.",
    }
    save_json(info, model_dir / "training_info.json")
    return info


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--config-dir", default="configs")
    ap.add_argument("--skip-predict", action="store_true")
    a = ap.parse_args()
    from ..data.load_data import load_processed
    cfgs = load_all_configs(a.config_dir)
    df = load_processed(cfgs["data"])
    meta = __import__("json").load(open(resolve("data/processed/preprocess_config.json")))
    info = train(cfgs, df, meta, max_epochs=a.epochs)
    print(f"[tft] trained: best_val_loss={info['best_val_loss']} epochs={info['epochs_run']} "
          f"time={info['train_seconds']:.0f}s params={info['n_parameters']}")
    if not a.skip_predict:
        from .predict import predict_and_save
        predict_and_save(cfgs, df, meta)


if __name__ == "__main__":
    main()
