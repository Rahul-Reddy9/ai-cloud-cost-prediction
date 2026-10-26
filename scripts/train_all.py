"""Train every learned component:  python scripts/train_all.py [--profile quick] [--epochs N] [--skip-tft]

Order: data -> TFT (+ forecasts) -> Isolation Forest (+ scores) -> PPO variants x seeds.
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd  # noqa: E402

from src.common import load_all_configs, resolve  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--profile", default="quick")
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--skip-tft", action="store_true")
    ap.add_argument("--skip-ppo", action="store_true")
    ap.add_argument("--config-dir", default="configs")
    a = ap.parse_args()
    cfgs = load_all_configs(a.config_dir)

    if not resolve("data/processed/preprocess_config.json").exists():
        from scripts.prepare_data import main as prep
        sys.argv = ["prepare_data"]
        prep()
    from src.data.load_data import load_processed
    df = load_processed(cfgs["data"])
    meta = json.load(open(resolve("data/processed/preprocess_config.json")))

    if not a.skip_tft:
        from src.forecasting.predict import predict_and_save
        from src.forecasting.train_tft import train
        info = train(cfgs, df, meta, max_epochs=a.epochs)
        print(f"[tft] val_loss={info['best_val_loss']} epochs={info['epochs_run']} {info['train_seconds']:.0f}s")
        predict_and_save(cfgs, df, meta)
    pred = pd.read_parquet(resolve(cfgs["forecasting"]["paths"]["predictions_file"]))

    from src.anomaly.score_anomalies import score_and_save
    from src.anomaly.train_detector import train_detectors
    train_detectors(cfgs, df, pred, meta)
    score_and_save(cfgs, df, pred, meta)

    if not a.skip_ppo:
        from src.controller.train_ppo import train_variant
        from src.evaluation.run_experiment import load_context
        ctx = load_context(cfgs)
        prof = cfgs["experiments"]["profiles"][a.profile]
        for v in cfgs["ppo"]["variants"]:
            for s in prof["seeds"]:
                info = train_variant(ctx, v, s, prof["total_timesteps"])
                print(f"[ppo] {v} seed{s}", "skipped" if info.get("skipped") else f"reward={info['best_selection_reward']:.4f}")


if __name__ == "__main__":
    main()
