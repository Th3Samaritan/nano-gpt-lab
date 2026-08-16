"""analyze.py - turn finished runs into tables + layman PNG reports.

WHAT
----
Read-only aggregation over runs/{dataset}/*/ (config.yaml + metrics.csv):
    python scripts/analyze.py --dataset shakespeare --size nano
    python scripts/analyze.py --dataset tinystories --size gpt2 --threshold 3.0
Writes results/{dataset}/comparison.json + the four-chart PNG suite:
    01_which_recipe_learned_best.png  - quality bar chart + winner star
    02_speed_and_memory.png           - efficiency bars
    03_learning_speed.png             - tokens-to-threshold bars
    04_verdict_card.png               - the 2x2 one-glance poster

WHY
----
Keeps reporting separate from training: you can re-analyze old runs after
the session ends (the runs dir is mirrored to Drive / persisted on
Kaggle), and the charts use the same code for the local machine and the T4 tiers.
"""
from __future__ import annotations

import argparse
import os
import sys

# Windows/conda OpenMP duplicate-runtime workaround (see scripts/train.py).
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.evaluation.analysis import analyze  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="analyze finished runs")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--size", default=None)
    parser.add_argument("--seeds", nargs="+", type=int, default=None)
    parser.add_argument("--threshold", type=float, default=2.0)
    args = parser.parse_args()

    report = analyze(args.dataset, args.size, args.seeds, args.threshold)
    print(report["table"])
    print("\nlayman reports:")
    for name, path in sorted(report["plots"].items()):
        print(f"  {name:<12} -> {path}")


if __name__ == "__main__":
    main()
