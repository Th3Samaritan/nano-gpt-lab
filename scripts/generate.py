"""generate.py - sample text from a trained run with fixed parameters.

WHAT
----
Loads a run's config + tokenizer + latest.pt and generates continuation
text. Default sampler settings come from the run's config (the SAME
settings used during training), so samples are comparable across variants;
--prompt / --num-tokens / --temperature / --top-k override them.

WHY
----
Plan section 27: generation must use identical parameters when comparing
samples, and hand-picking pretty outputs is banned - so this script writes
the raw sample verbatim and you compare the files, not your memory of
which one looked nicer.

WHERE
----
    python scripts/generate.py --run-dir runs/shakespeare/rope_flash_fused_nano_seed42
    python scripts/generate.py --run-dir ... --prompt "To be, or not to be" --num-tokens 300
"""
from __future__ import annotations

import argparse
import os
import sys

# Windows/conda OpenMP duplicate-runtime workaround (see scripts/train.py).
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.evaluation.generation import generate_text  # noqa: E402
from src.model.gpt import GPT, GPTConfig  # noqa: E402
from src.tokenizer.bpe import BPETokenizer  # noqa: E402
from src.utils.checkpoint import load_checkpoint  # noqa: E402
from src.utils.config import load_yaml  # noqa: E402
from src.utils.device import get_device  # noqa: E402
from src.utils.environment import repo_root  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="generate from a trained run")
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--prompt", default=None)
    parser.add_argument("--num-tokens", type=int, default=None)
    parser.add_argument("--temperature", type=float, default=None)
    parser.add_argument("--top-k", type=int, default=None)
    args = parser.parse_args()

    cfg = load_yaml(os.path.join(args.run_dir, "config.yaml"))
    device = get_device()
    model_cfg = GPTConfig(
        block_size=cfg["block_size"], vocab_size=cfg["vocab_size"],
        n_layer=cfg["n_layer"], n_head=cfg["n_head"], n_embd=cfg["n_embd"],
        dropout=cfg.get("dropout", 0.0), bias=cfg.get("bias", True),
        attention=cfg["attention"], position=cfg["position"],
        rope_base=cfg.get("rope_base", 10000.0),
        flash_tile_size=cfg.get("flash_tile_size", 64))
    model = GPT(model_cfg).to(device)
    load_checkpoint(os.path.join(args.run_dir, "latest.pt"), model)

    tokenizer_path = os.path.join(args.run_dir, "tokenizer.json")
    if not os.path.exists(tokenizer_path):
        tokenizer_path = os.path.join(os.path.dirname(args.run_dir), "..", "..",
                                      "data", cfg["dataset"], "tokenizer.json")
        tokenizer_path = os.path.abspath(tokenizer_path)
    tokenizer = BPETokenizer.load(tokenizer_path)

    eval_cfg = cfg.get("eval", {})
    prompt = args.prompt or eval_cfg.get("prompt", "The ")
    text = generate_text(
        model, tokenizer, prompt,
        temperature=args.temperature if args.temperature is not None
        else eval_cfg.get("temperature", 0.8),
        top_k=args.top_k if args.top_k is not None
        else eval_cfg.get("top_k", 40),
        top_p=eval_cfg.get("top_p"),
        max_new_tokens=args.num_tokens or eval_cfg.get("max_new_tokens", 200))
    print(f"prompt   : {prompt!r}")
    print(f"config   : {cfg.get('attention')} + {cfg.get('position')} "
          f"({cfg.get('size')}, seed {cfg.get('seed')})")
    print("-" * 72)
    print(text)
    out_path = os.path.join(repo_root(), "results", "samples",
                            os.path.basename(args.run_dir) + ".txt")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(f"run: {args.run_dir}\nprompt: {prompt!r}\n\n{text}\n")
    print(f"\nsaved -> {out_path}")


if __name__ == "__main__":
    main()
