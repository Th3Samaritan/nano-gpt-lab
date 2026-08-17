"""evaluate.py - the industry-standard evaluation battery for a finished run.

WHAT
----
Loads a run's config + tokenizer + latest.pt and produces eval_report.json
inside the run dir:

    val_loss, val_ppl, bits_per_byte
    ece                       - calibration: does confidence match accuracy?
    length_buckets            - long-range penalty: loss by context position
    few_shot                  - ICL accuracy at 0/1/3/5 shots
    generation                - distinct-1/2 + repetition rate of samples
    inference_tokens_per_sec  - deployment-facing generation speed

WHY
----
One evaluation function for every experiment (plan section 27), extended
to the professional metric battery: loss alone cannot show a model that is
miscalibrated, repetitive, or weak at long context. Everything is
deterministic (same items, same negatives) so variants are comparable.

WHERE
----
    python scripts/evaluate.py --run-dir runs/shakespeare/flash_fused_rope_gelu_adamw_nano_seed42
    python scripts/evaluate.py --run-dir ... --shots "0 1 3 5" --n-items 40
"""
from __future__ import annotations

import argparse
import json
import os
import sys

# Windows/conda OpenMP duplicate-runtime workaround (see scripts/train.py).
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch  # noqa: E402

from src.evaluation.generation import generate_text  # noqa: E402
from src.evaluation.loss import estimate_loss  # noqa: E402
from src.evaluation.metrics import (  # noqa: E402
    bits_per_byte, collect_confidence_samples, distinct_n,
    expected_calibration_error, inference_tokens_per_sec,
    loss_by_position_bucket, repetition_rate)
from src.evaluation.perplexity import perplexity_from_loss  # noqa: E402
from src.evaluation.shots import build_icl_task, rank_accuracy  # noqa: E402
from src.model.gpt import GPT, GPTConfig  # noqa: E402
from src.tokenizer.bpe import BPETokenizer  # noqa: E402
from src.training.precision import PrecisionContext  # noqa: E402
from src.training.trainer import TokenDataLoader, _ensure_data  # noqa: E402
from src.utils.checkpoint import load_checkpoint  # noqa: E402
from src.utils.config import load_yaml  # noqa: E402
from src.utils.device import get_autocast_dtype, get_device  # noqa: E402
from src.utils.environment import runs_dir  # noqa: E402


def _tokens_per_char(tokenizer: BPETokenizer, text: str) -> float:
    """tokens/chars over a sample - needed to convert loss to bpb."""
    sample = text[:1_000_000]
    ids = tokenizer.encode(sample)
    return len(ids) / max(len(sample), 1)


def main() -> None:
    parser = argparse.ArgumentParser(description="full evaluation of a run")
    parser.add_argument("--run-dir", default=None)
    parser.add_argument("--dataset", default=None)
    parser.add_argument("--attention", default=None)
    parser.add_argument("--position", default=None)
    parser.add_argument("--size", default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--shots", nargs="+", type=int, default=[0, 1, 3, 5],
                        help="n-shot settings for the ICL test")
    parser.add_argument("--n-items", type=int, default=40,
                        help="ICL test items")
    parser.add_argument("--n-samples", type=int, default=4,
                        help="generation samples for diversity metrics")
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
        activation=cfg.get("activation", "gelu"),
        swiglu_matched=cfg.get("swiglu_matched", True),
        rope_base=cfg.get("rope_base", 10000.0),
        flash_tile_size=cfg.get("flash_tile_size", 64))
    model = GPT(model_cfg).to(device)
    load_checkpoint(os.path.join(run_dir, "latest.pt"), model)
    precision = PrecisionContext(device, get_autocast_dtype(device))

    tokenizer_path = os.path.join(run_dir, "tokenizer.json")
    if not os.path.exists(tokenizer_path):
        tokenizer_path = data["tokenizer_path"]
    tokenizer = BPETokenizer.load(tokenizer_path)

    loader = TokenDataLoader(data["val_bin"], cfg["block_size"],
                             cfg["micro_batch_size"],
                             data.get("token_dtype", "uint16"))

    report = {"run_dir": run_dir}

    # ---- 1. core quality -------------------------------------------------
    loss = estimate_loss(model, loader.get_batch, precision, 50)
    report["val_loss"] = round(loss, 4)
    report["val_ppl"] = round(perplexity_from_loss(loss), 2)
    with open(os.path.join(os.path.dirname(data["val_bin"]), "val.txt"),
              "r", encoding="utf-8", errors="replace") as f:
        tpc = _tokens_per_char(tokenizer, f.read())
    report["bits_per_byte"] = round(bits_per_byte(loss, tpc), 4)

    # ---- 2. calibration ---------------------------------------------------
    confs, corrects = collect_confidence_samples(
        model, loader.get_batch, precision, n_batches=10)
    report["ece"] = round(expected_calibration_error(confs, corrects), 4)

    # ---- 3. long-range penalty -------------------------------------------
    report["length_buckets"] = [
        round(v, 4) for v in loss_by_position_bucket(
            model, loader.get_batch, precision)]

    # ---- 4. zero/one/few-shot in-context learning -------------------------
    with open(os.path.join(os.path.dirname(data["val_bin"]), "val.txt"),
              "r", encoding="utf-8", errors="replace") as f:
        val_text = f.read()
    items, examples = build_icl_task(val_text, n_items=args.n_items)
    report["few_shot"] = {
        str(k): round(rank_accuracy(model, tokenizer, items, examples,
                                    precision, n_shots=k), 4)
        for k in args.shots}

    # ---- 5. generation diversity -----------------------------------------
    eval_cfg = cfg.get("eval", {})
    prompt = eval_cfg.get("prompt", "The ")
    gen_kwargs = {
        "temperature": eval_cfg.get("temperature", 0.8),
        "top_k": eval_cfg.get("top_k", 40),
        "top_p": eval_cfg.get("top_p", None),
        "max_new_tokens": eval_cfg.get("max_new_tokens", 200),
    }
    samples = [generate_text(model, tokenizer, prompt, **gen_kwargs)
               for _ in range(args.n_samples)]
    samples_ids = [tokenizer.encode(s) for s in samples]
    report["generation"] = {
        "distinct_1": round(float(sum(distinct_n(t, 1)
                                      for t in samples_ids)) / len(samples_ids), 4),
        "distinct_2": round(float(sum(distinct_n(t, 2)
                                      for t in samples_ids)) / len(samples_ids), 4),
        "repetition_rate": round(float(sum(repetition_rate(t)
                                           for t in samples_ids)) / len(samples_ids), 4),
    }

    # ---- 6. deployment-facing inference speed ----------------------------
    report["inference_tokens_per_sec"] = round(
        inference_tokens_per_sec(model, tokenizer, prompt), 1)

    out_path = os.path.join(run_dir, "eval_report.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(json.dumps(report, indent=2))
    print(f"saved -> {out_path}")


if __name__ == "__main__":
    main()
