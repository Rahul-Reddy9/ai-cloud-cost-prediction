"""Run Experiments 1-5, then build tables/figures/report:
   python scripts/run_all_experiments.py [--profile quick|full] [--only exp4] [--force-train]"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.evaluation.ablations import run_ablations  # noqa: E402
from src.evaluation.compare_results import build_all  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--profile", default="quick")
    ap.add_argument("--only", default=None, choices=[None, "exp1", "exp2", "exp3", "exp4", "exp5"])
    ap.add_argument("--force-train", action="store_true")
    a = ap.parse_args()
    run_ablations(a.profile, a.only, a.force_train)
    build_all()
    try:
        from src.evaluation.report import write_report
        write_report()
    except ImportError:
        pass


if __name__ == "__main__":
    main()
