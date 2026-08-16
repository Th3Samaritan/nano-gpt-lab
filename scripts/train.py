"""train.py - the main training entry point (one codebase, all platforms).

WHAT
----
Runs one experiment (or the full A/B/C/D comparison) on one dataset:
    python scripts/train.py --dataset shakespeare --exp exp_a_vanilla_learned
    python scripts/train.py --dataset shakespeare --compare --size nano
    python scripts/train.py --dataset tinystories --compare --size gpt2 --seeds 42 123
    python scripts/train.py ... --steps 200   (quick override for debugging)
    python scripts/train.py ... --resume      (continue from latest.pt)

WHERE
----
local-machine tier : --size nano (default) - fast, CPU-friendly.
Colab/Kaggle T4  : --size gpt2 (or small) - the scaled tier.
Outputs: runs/{dataset}/{attention}_{position}_{size}_{seed}/ and, after a
comparison, results/{dataset}/ with the layman PNG suite + comparison.json.
"""
from __future__ import annotations

import argparse
import os
import sys

# Windows + conda sometimes ships two OpenMP runtimes (torch's and numpy/
# MKL's); loading both aborts with "OMP: Error #15". This is the standard
# workaround for that specific clash and must be set BEFORE torch/numpy load.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.evaluation.analysis import analyze  # noqa: E402
from src.training.trainer import train  # noqa: E402
from src.utils.config import build_run_config  # noqa: E402
from src.utils.environment import print_environment, sync_run_to_drive  # noqa: E402

EXPERIMENTS = {
    "exp_a_vanilla_learned": ("A", "vanilla", "learned"),
    "exp_b_vanilla_rope": ("B", "vanilla", "rope"),
    "exp_c_flash_learned": ("C", "flash_fused", "learned"),
    "exp_d_flash_rope": ("D", "flash_fused", "rope"),
}


def main() -> None:
    parser = argparse.ArgumentParser(description="nano-gpt-lab training")
    parser.add_argument("--dataset", default="shakespeare",
                        choices=["shakespeare", "tinystories", "openwebtext"])
    parser.add_argument("--exp", default=None,
                        choices=list(EXPERIMENTS),
                        help="single experiment; omit with --compare")
    parser.add_argument("--compare", action="store_true",
                        help="run the full A/B/C/D matrix")
    parser.add_argument("--size", default=None,
                        choices=["nano", "small", "gpt2", "gpt3"],
                        help="model size preset (default: dataset's)")
    parser.add_argument("--seeds", nargs="+", type=int, default=[42],
                        help="seeds to run (plan: 42 123 2026)")
    parser.add_argument("--steps", type=int, default=None,
                        help="override max_steps (quick experiments)")
    parser.add_argument("--devices", default="auto",
                        help="GPUs to use: 'auto' (all), 1, or 2 (Kaggle T4x2)")
    parser.add_argument("--resume", action="store_true",
                        help="resume each run from its latest.pt")
    parser.add_argument("--threshold", type=float, default=2.0,
                        help="val-loss threshold for the learning-speed chart")
    args = parser.parse_args()

    print_environment()
    experiments = list(EXPERIMENTS) if args.compare else [args.exp]
    if args.exp is None and not args.compare:
        parser.error("provide --exp <name> or --compare")

    from src.utils.config import load_dataset_config
    resolved_size = args.size or load_dataset_config(args.dataset).get("size", "nano")

    all_summaries = []
    for exp_name in experiments:
        overrides = {}
        if args.steps is not None:
            overrides["max_steps"] = args.steps
        try:
            overrides["devices"] = int(args.devices)
        except ValueError:
            overrides["devices"] = args.devices  # "auto"
        # Resolve the size preset: explicit --size wins, else the dataset's
        # yaml default (nano for shakespeare, gpt2 for the T4-tier datasets).
        for seed in args.seeds:
            cfg = build_run_config(args.dataset, exp_name, size=resolved_size,
                                   overrides=overrides)
            cfg["seed"] = seed
            summary = train(cfg, resume=args.resume)
            all_summaries.append(summary)

    if len(all_summaries) > 1 or args.compare:
        from src.evaluation.analysis import render_layman_reports, \
            save_comparison_json, comparison_table
        from src.utils.environment import repo_root
        out_dir = os.path.join(repo_root(), "results", args.dataset, resolved_size)
        save_comparison_json(all_summaries,
                             os.path.join(out_dir, "comparison.json"))
        render_layman_reports(all_summaries, args.dataset, out_dir,
                              args.threshold)
        print("\n" + comparison_table(all_summaries))
        print(f"\nlayman reports -> {out_dir}")
        sync_run_to_drive(os.path.dirname(out_dir), full=False)


if __name__ == "__main__":
    main()
