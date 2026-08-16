"""evaluate.py - score a finished (or resumed) run on its validation split.

WHAT
----
Loads config.yaml + tokenizer + latest.pt from a run directory, rebuilds
the validation data loader, and reports loss + perplexity with the SAME
estimator the trainer uses (evaluation/loss.py). Never re-trains.

WHY
----
A unified evaluation function (plan section 27) means the numbers printed
here are directly comparable to the numbers logged during training, and
across variants.

WHERE
----
    python scripts/evaluate.py --dataset shakespeare --attention vanilla --position learned --size nano --seed 42
    python scripts/evaluate.py --run-dir runs/shakespeare/vanilla_learned_nano_seed42
"""
from __future__ import annotations

import argparse
import json
import os
import sys

# Windows/conda OpenMP duplicate-runtime workaround (see scripts/train.py).
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.evaluation.loss import estimate_loss  # noqa: E402
from src.evaluation.perplexity import perplexity_from_loss  # noqa: E402
from src.model.gpt import GPT, GPTConfig  # noqa: E402
from src.training.precision import PrecisionContext  # noqa: E402
from src.training.trainer import TokenDataLoader, _ensure_data  # noqa: E402
from src.utils.checkpoint import load_checkpoint  # noqa: E402
from src.utils.config import load_yaml  # noqa: E402
from src.utils.device import get_autocast_dtype, get_device  # noqa: E402
from src.utils.environment import runs_dir  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="evaluate a finished run")
    parser.add_argument("--run-dir", default=None)
    parser.add_argument("--dataset", default=None)
    parser.add_argument("--attention", default=None)
    parser.add_argument("--position", default=None)
    parser.add_argument("--size", default=None)
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()

    if args.run_dir:
        run_dir = args.run_dir
    else:
        if not all([args.dataset, args.attention, args.position, args.size,
                    args.seed is not None]):
            parser.error("give --run-dir or all of --dataset --attention "
                         "--position --size --seed")
        run_dir = os.path.join(
            runs_dir(), args.dataset,
            f"{args.attention}_{args.position}_{args.size}_seed{args.seed}")

    cfg_path = os.path.join(run_dir, "config.yaml")
    if not os.path.exists(cfg_path):
        sys.exit(f"no config.yaml in {run_dir} - not a trained run?")
    cfg = load_yaml(cfg_path)

    device = get_device()
    data = _ensure_data(cfg)
    model_cfg = GPTConfig(
        block_size=cfg["block_size"], vocab_size=cfg["vocab_size"],
        n_layer=cfg["n_layer"], n_head=cfg["n_head"], n_embd=cfg["n_embd"],
        dropout=cfg.get("dropout", 0.0), bias=cfg.get("bias", True),
        attention=cfg["attention"], position=cfg["position"],
        rope_base=cfg.get("rope_base", 10000.0),
        flash_tile_size=cfg.get("flash_tile_size", 64))
    model = GPT(model_cfg).to(device)
    load_checkpoint(os.path.join(run_dir, "latest.pt"), model)
    precision = PrecisionContext(device, get_autocast_dtype(device))

    loader = TokenDataLoader(data["val_bin"], cfg["block_size"],
                             cfg["micro_batch_size"],
                             data.get("token_dtype", "uint16"))
    loss = estimate_loss(model, loader.get_batch, precision, 100)
    ppl = perplexity_from_loss(loss)
    print(json.dumps({
        "run_dir": run_dir,
        "val_loss": round(loss, 4),
        "val_ppl": round(ppl, 2),
        "params": model.get_num_params(non_embedding=False),
    }, indent=2))


if __name__ == "__main__":
    main()
