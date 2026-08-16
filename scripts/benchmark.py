"""benchmark.py - memory/throughput scan to pick a feasible batch/context.

WHAT
----
For a given experiment + size, scans the micro_batch x block_size grid
(plan section 14's pattern: 4 -> 2 -> 1 if necessary; 1024 -> 512 -> 256),
measuring for each combination:
    - forward+backward step time (average of a few steps)
    - tokens/sec throughput
    - peak GPU memory
and prints a feasibility table with a recommendation.

WHY
----
The plan explicitly says batch/context values should be "determined by a
memory benchmark rather than assumed". A T4 has 16GB; the gpt2 preset at
micro_batch 4 / block 1024 is a sensible start but this script is how you
PROVE it before a long run.

WHERE
----
    python scripts/benchmark.py --dataset shakespeare --exp exp_d_flash_rope --size gpt2
Run it on the T4 before starting scaled training; on the ZBook it gives
CPU-relative timings (memory column is 0 without CUDA).
"""
from __future__ import annotations

import argparse
import os
import sys
import time

# Windows/conda OpenMP duplicate-runtime workaround (see scripts/train.py).
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch  # noqa: E402

from src.model.gpt import GPT, GPTConfig  # noqa: E402
from src.training.precision import PrecisionContext  # noqa: E402
from src.training.trainer import TokenDataLoader, _ensure_data  # noqa: E402
from src.utils.config import build_run_config, load_dataset_config  # noqa: E402
from src.utils.device import get_autocast_dtype, get_device, peak_memory_mb, \
    reset_peak_memory  # noqa: E402


def bench_combo(model_cfg, precision, batch: int, block: int, data: dict,
                device: str, steps: int = 4):
    """Time avg step + peak memory for one (micro_batch, block) combo."""
    model = GPT(model_cfg).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    loader = TokenDataLoader(data["train_bin"], block, batch,
                             data.get("token_dtype", "uint16"))
    reset_peak_memory()
    torch.cuda.empty_cache() if device == "cuda" else None
    times = []
    try:
        for _ in range(steps):
            x, y = loader.get_batch()
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad(set_to_none=True)
            t0 = time.time()
            with precision.autocast_ctx:
                _, loss = model(x, y)
            precision.backward(loss)
            optimizer.step()
            if device == "cuda":
                torch.cuda.synchronize()
            times.append(time.time() - t0)
    except torch.cuda.OutOfMemoryError:
        return None  # combination does not fit
    dt = sum(times) / len(times)
    tok_per_sec = batch * block / dt
    return {"step_time_s": round(dt, 3), "tokens_per_sec": round(tok_per_sec, 1),
            "peak_mem_mb": round(peak_memory_mb(), 1)}


def main() -> None:
    parser = argparse.ArgumentParser(description="batch/context feasibility scan")
    parser.add_argument("--dataset", default="shakespeare")
    parser.add_argument("--exp", default="exp_d_flash_rope")
    parser.add_argument("--size", default=None)
    parser.add_argument("--batches", nargs="+", type=int, default=[1, 2, 4, 8])
    parser.add_argument("--blocks", nargs="+", type=int, default=[256, 512, 1024])
    args = parser.parse_args()

    size = args.size or load_dataset_config(args.dataset).get("size", "nano")
    cfg = build_run_config(args.dataset, args.exp, size=size)
    device = get_device()
    precision = PrecisionContext(device, get_autocast_dtype(device))
    data = _ensure_data(cfg)
    model_cfg = GPTConfig(
        block_size=max(args.blocks), vocab_size=cfg["vocab_size"],
        n_layer=cfg["n_layer"], n_head=cfg["n_head"], n_embd=cfg["n_embd"],
        dropout=cfg.get("dropout", 0.0), bias=cfg.get("bias", True),
        attention=cfg["attention"], position=cfg["position"],
        rope_base=cfg.get("rope_base", 10000.0),
        flash_tile_size=cfg.get("flash_tile_size", 64))

    print(f"scan: {args.exp} @ {size} | device={device}")
    print(f"{'micro_batch':>11} {'block':>6} {'step_s':>8} {'tok/s':>9} "
          f"{'peakMB':>8}")
    results = []
    for block in args.blocks:
        for batch in args.batches:
            r = bench_combo(model_cfg, precision, batch, block, data, device)
            tag = "" if r else "OOM"
            print(f"{batch:>11} {block:>6} "
                  f"{r['step_time_s'] if r else '':>8} "
                  f"{r['tokens_per_sec'] if r else '':>9} "
                  f"{r['peak_mem_mb'] if r else '':>8} {tag}")
            if r:
                results.append((batch, block, r))
    if results:
        best = max(results, key=lambda t: t[2]["tokens_per_sec"])
        b, blk, r = best
        print(f"\nrecommendation: micro_batch {b}, block {blk} "
              f"({r['tokens_per_sec']} tok/s, {r['peak_mem_mb']} MB)")
        print("put these in your config or pass --steps after adjusting.")


if __name__ == "__main__":
    main()
