"""Validate and prepare the dataset:  python scripts/prepare_data.py [--regenerate] [--config-dir configs]"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.common import load_all_configs, save_json  # noqa: E402
from src.data.load_data import load_raw  # noqa: E402
from src.data.preprocess import preprocess, save_processed  # noqa: E402
from src.data.validate_data import validate_dataframe  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--regenerate", action="store_true", help="regenerate the synthetic dataset")
    ap.add_argument("--config-dir", default="configs")
    a = ap.parse_args()
    cfgs = load_all_configs(a.config_dir)
    df = load_raw(cfgs["data"], cfgs["simulation"], regenerate=a.regenerate)
    rep = validate_dataframe(df, cfgs["data"])
    print(f"[validate] ok={rep.ok}  errors={len(rep.errors)}  warnings={len(rep.warnings)}")
    for w in rep.warnings:
        print("  warn:", w)
    rep.raise_if_errors()
    out, meta = preprocess(df, cfgs["data"])
    save_processed(out, meta, cfgs["data"])
    save_json(rep.to_dict(), "results/metrics/data_validation.json")
    print(f"[preprocess] rows={len(out)} groups={meta['n_groups']} imputed={meta['rows_imputed']} "
          f"dropped={meta['rows_dropped_long_gaps']} split={meta['rows_per_split']}")
    print("[preprocess] saved data/processed/ (parquet + preprocess_config.json)")


if __name__ == "__main__":
    main()
