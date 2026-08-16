"""The training loop: micro-batches, grad accumulation, eval, checkpoints.

WHAT
----
train() drives one full run from a merged config dict (see
src/utils/config.py). The loop:

    per optimizer step (gradient_accumulation_steps micro-batches):
        sample a micro-batch (block_size tokens each)
        forward under mixed precision -> cross-entropy loss
        backward (scaled in fp16)
    clip gradients -> optimizer step -> LR scheduler step

    every eval_interval steps:
        estimate train + val loss (evaluation/loss.py), log metrics.csv,
        record throughput + peak VRAM, plot loss curves
    every sample_interval steps:
        generate with FIXED sampler settings, append to samples.txt
    every checkpoint_interval steps (and at the end):
        save latest.pt (+ best.pt when val loss improves); on Colab,
        mirror checkpoints to Drive (plan section 24: train on runtime
        storage, copy to Drive periodically)

WHY
----
Every A/B/C/D run enters through this SAME loop with the SAME recipe, so
throughput/loss/memory differences are caused by the attention backend and
position scheme alone. Resume restores optimizer momentum, LR schedule,
scaler and RNG state - an interrupted run continues bit-identically.

WHERE
-----
Called by scripts/train.py (one call per seed/variant); artifacts land in
runs/{dataset}/{attention}_{position}_{size}_{seed}/.
"""
from __future__ import annotations

import os
import shutil
import time
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F

from src.model.gpt import GPT, GPTConfig
from src.evaluation.generation import generate_text
from src.evaluation.loss import estimate_loss
from src.evaluation.perplexity import perplexity_from_loss
from src.training.optimizer import create_optimizer
from src.training.precision import PrecisionContext
from src.training.scheduler import CosineWarmupScheduler
from src.tokenizer.bpe import BPETokenizer
from src.utils.checkpoint import (
    find_latest_checkpoint, load_checkpoint, save_checkpoint)
from src.utils.config import save_yaml
from src.utils.device import (
    get_autocast_dtype, get_base_gpt, get_device, gpu_peak_flops,
    peak_memory_mb, reset_peak_memory, resolve_device_ids, wrap_parallel)
from src.utils.environment import (
    runs_dir, sync_run_to_drive, write_environment_file)
from src.utils.logging import RunLogger, plot_loss_curves
from src.utils.seed import seed_all


# --------------------------------------------------------------------------
# Token data loader (memmap over the .bin caches, nanoGPT style)
# --------------------------------------------------------------------------
class TokenDataLoader:
    """Samples (x, y) pairs - y is x shifted by one token (next-token LM)."""

    def __init__(self, bin_path: str, block_size: int, batch_size: int,
                 token_dtype: str = "uint16"):
        # Read-only memmap: the whole token array stays on disk, pages are
        # loaded on demand - this is what makes 1GB+ token caches usable.
        self.data = np.memmap(
            bin_path, dtype=np.uint16 if token_dtype == "uint16" else np.uint32,
            mode="r")
        self.block_size = block_size
        self.batch_size = batch_size

    def get_batch(self) -> tuple:
        """Random contiguous windows: x = tokens[i:i+T], y = tokens[i+1:i+T+1]."""
        ix = torch.randint(0, len(self.data) - self.block_size, (self.batch_size,))
        x = torch.stack([
            torch.from_numpy(self.data[i:i + self.block_size].astype(np.int64))
            for i in ix])
        y = torch.stack([
            torch.from_numpy(self.data[i + 1:i + 1 + self.block_size].astype(np.int64))
            for i in ix])
        return x, y


# --------------------------------------------------------------------------
# Run directory + naming (plan section 22)
# --------------------------------------------------------------------------
def run_dir_for(config: dict) -> str:
    """runs/{dataset}/{attention}_{position}_{size}_{seed}"""
    return os.path.join(
        runs_dir(), config["dataset"],
        f"{config['attention']}_{config['position']}_{config['size']}_seed{config['seed']}")


# --------------------------------------------------------------------------
# Data preparation entry (lazy: prepares only if the cache is missing)
# --------------------------------------------------------------------------
def _ensure_data(config: dict) -> dict:
    """Prepare the dataset once (download -> BPE -> tokenize) and return
    the paths needed for training."""
    from data.prepare_shakespeare import prepare as prep_shakespeare
    from data.prepare_tinystories import prepare as prep_tinystories
    from data.prepare_openwebtext import prepare as prep_openwebtext
    preparers = {
        "shakespeare": prep_shakespeare,
        "tinystories": prep_tinystories,
        "openwebtext": prep_openwebtext,
    }
    if config["dataset"] not in preparers:
        raise KeyError(f"unknown dataset '{config['dataset']}'")
    return preparers[config["dataset"]](config)


# --------------------------------------------------------------------------
# Main training function
# --------------------------------------------------------------------------
def train(config: dict, resume: bool = False) -> dict:
    """Run one training (one variant, one seed) to completion.

    Returns the summary dict (final/best val loss, ppl, tokens/sec, peak
    VRAM, params, run dir) that scripts/train.py aggregates for the
    A/B/C/D comparison table.
    """
    device = get_device()
    dtype = get_autocast_dtype(device)
    # Multi-GPU (Kaggle T4x2): resolve ONCE, record it in the saved config
    # so the run is reproducible, and wrap the model AFTER the optimizer is
    # built (DataParallel replicas share the base parameters - the
    # optimizer must hold references to the base module's params).
    device_ids = resolve_device_ids(config.get("devices", "auto"))
    config = dict(config)
    config["devices_used"] = [int(d) for d in device_ids] if device_ids else ["cpu"]
    run_dir = run_dir_for(config)
    os.makedirs(run_dir, exist_ok=True)

    # ---- reproducibility bookkeeping (plan section 7) ------------------
    save_yaml(config, os.path.join(run_dir, "config.yaml"))
    write_environment_file(run_dir)
    seed_all(config["seed"])
    logger = RunLogger(run_dir, resume=resume)

    # ---- data + tokenizer (shared by all four variants) ----------------
    data = _ensure_data(config)
    tokenizer = BPETokenizer.load(data["tokenizer_path"])
    for name in ("tokenizer.json", "meta.json"):
        src = os.path.join(os.path.dirname(data["tokenizer_path"]), name)
        if os.path.exists(src):
            shutil.copy2(src, os.path.join(run_dir, name))

    # ---- model (architecture comes ONLY from the merged config) --------
    model_cfg = GPTConfig(
        block_size=config["block_size"],
        vocab_size=config["vocab_size"],
        n_layer=config["n_layer"],
        n_head=config["n_head"],
        n_embd=config["n_embd"],
        dropout=config.get("dropout", 0.0),
        bias=config.get("bias", True),
        attention=config["attention"],
        position=config["position"],
        rope_base=config.get("rope_base", 10000.0),
        flash_tile_size=config.get("flash_tile_size", 64),
    )
    model = GPT(model_cfg).to(device)
    with open(os.path.join(run_dir, "model_summary.txt"), "w", encoding="utf-8") as f:
        f.write(model.model_summary() + "\n")
    logger.log(model.model_summary().replace("\n", " | "))

    # ---- optimizer / scheduler / precision ------------------------------
    # Built from the BASE model before wrapping (see device_ids note above).
    optimizer = create_optimizer(
        model, config.get("weight_decay", 0.1),
        config["learning_rate"], tuple(config.get("betas", [0.9, 0.95])), device)
    scheduler = CosineWarmupScheduler(
        optimizer, config.get("warmup_steps", 100),
        config.get("decay_steps") or config["max_steps"],
        config.get("min_learning_rate", config["learning_rate"] * 0.1))
    precision = PrecisionContext(device, dtype)
    model, base_gpt = wrap_parallel(model, device_ids)

    # ---- resume (plan section 26: full state, exact continuation) -------
    start_step, tokens_seen, best_val_loss, last_val_loss = 0, 0, float("inf"), float("inf")
    if resume:
        latest = find_latest_checkpoint(run_dir)
        if latest is not None:
            state = load_checkpoint(latest, base_gpt, optimizer, scheduler, precision)
            start_step, tokens_seen = state["step"], state["tokens_seen"]
            best_val_loss = state.get("best_val_loss", float("inf"))
            logger.log(f"resumed from {os.path.basename(latest)} at step {start_step}")
    scheduler.step_count = start_step  # scheduler is stateless: sync the counter

    # ---- data loaders ----------------------------------------------------
    micro = config["micro_batch_size"]
    accum = config["gradient_accumulation_steps"]
    block = config["block_size"]
    eff_tokens = micro * accum * block
    token_dtype = data.get("token_dtype", "uint16")
    train_loader = TokenDataLoader(data["train_bin"], block, micro, token_dtype)
    val_loader = TokenDataLoader(data["val_bin"], block, micro, token_dtype)
    logger.log(
        f"micro_batch={micro} accum={accum} effective_tokens/step={eff_tokens} "
        f"precision={dtype or 'fp32'} device={device}"
        f"{' x' + str(len(device_ids)) if len(device_ids) > 1 else ''}")

    # ---- eval / sample parameters (identical for every variant) ---------
    eval_cfg = config.get("eval", {})
    prompt = eval_cfg.get("prompt", "The ")
    gen_kwargs = {
        "temperature": eval_cfg.get("temperature", 0.8),
        "top_k": eval_cfg.get("top_k", 40),
        "top_p": eval_cfg.get("top_p", None),
        "max_new_tokens": eval_cfg.get("max_new_tokens", 200),
    }

    # ---- the training loop ----------------------------------------------
    max_steps = config["max_steps"]
    eval_interval = config.get("eval_interval", 100)
    checkpoint_interval = config.get("checkpoint_interval", 500)
    sample_interval = config.get("sample_interval", 500)
    grad_clip = config.get("grad_clip", 1.0)
    model.train()
    reset_peak_memory()
    t_total0 = time.time()

    for step in range(start_step, max_steps):
        t0 = time.time()
        optimizer.zero_grad(set_to_none=True)
        lossf = 0.0
        for _ in range(accum):
            x, y = train_loader.get_batch()
            x, y = x.to(device), y.to(device)
            with precision.autocast_ctx:
                # Loss computed OUTSIDE the model: the plain GPT returns
                # (logits, None) and the DataParallel wrapper returns
                # gathered logits - one code path, one semantics, so
                # multi-GPU runs are comparable with single-GPU runs.
                out = model(x)
                logits = out[0] if isinstance(out, tuple) else out
                loss = F.cross_entropy(
                    logits.view(-1, logits.size(-1)), y.view(-1))
            loss = loss / accum          # mean over the effective batch
            precision.backward(loss)
            lossf += loss.item() * accum
        grad_norm = precision.optimizer_step(optimizer, grad_clip)
        scheduler.step()
        dt = time.time() - t0
        tokens_seen += eff_tokens

        # ---- periodic evaluation ----------------------------------------
        if (step + 1) % eval_interval == 0 or step == start_step:
            train_loss = estimate_loss(model, train_loader.get_batch,
                                       precision, config.get("eval_iters", 20))
            val_loss = estimate_loss(model, val_loader.get_batch,
                                     precision, config.get("eval_iters", 20))
            last_val_loss = val_loss
            best_val_loss = min(best_val_loss, val_loss)
            elapsed = time.time() - t_total0
            tok_per_sec = tokens_seen / max(elapsed, 1e-6)
            n_gpus = max(len(device_ids), 1)
            mfu = (base_gpt.flops_per_token() * eff_tokens / max(dt, 1e-6)
                   / (gpu_peak_flops(device) * n_gpus)) if device == "cuda" else 0.0
            row = {
                "step": step + 1, "tokens_seen": tokens_seen,
                "train_loss": round(train_loss, 4),
                "val_loss": round(val_loss, 4),
                "val_ppl": round(perplexity_from_loss(val_loss), 2),
                "lr": round(scheduler.current_lr, 8),
                "step_time_s": round(dt, 3),
                "tokens_per_sec": round(tok_per_sec, 1),
                "gpu_mem_mb": round(peak_memory_mb(), 1),
                "grad_norm": round(grad_norm, 4),
            }
            logger.record_metrics(row)
            logger.log(
                f"step {step+1}/{max_steps} | train {train_loss:.3f} "
                f"val {val_loss:.3f} ppl {row['val_ppl']} | {tok_per_sec:.0f} "
                f"tok/s | lr {scheduler.current_lr:.2e} | "
                f"gpu {row['gpu_mem_mb']:.0f}MB | mfu {mfu*100:.1f}%")
            plot_loss_curves(run_dir)
            sync_run_to_drive(run_dir, full=False)

        # ---- periodic generation (fixed sampler settings) ----------------
        if sample_interval > 0 and (step + 1) % sample_interval == 0:
            text = generate_text(model, tokenizer, prompt, **gen_kwargs)
            with open(os.path.join(run_dir, "samples.txt"), "a", encoding="utf-8") as f:
                f.write(f"\n===== step {step+1} =====\n{text}\n")
            logger.log(f"sample generated at step {step+1}")

        # ---- periodic checkpointing (+ Drive mirror on Colab) -----------
        if (step + 1) % checkpoint_interval == 0 or (step + 1) == max_steps:
            # Checkpoints store the BASE model (no DataParallel 'module.'
            # prefixes), so any run loads them regardless of GPU count.
            save_checkpoint(
                os.path.join(run_dir, "latest.pt"), base_gpt, optimizer,
                scheduler, precision, step + 1, tokens_seen, config,
                best_val_loss, is_best=(best_val_loss == last_val_loss))
            sync_run_to_drive(run_dir, full=True)
            logger.log(f"checkpoint saved at step {step+1}")

    # ---- final artifacts --------------------------------------------------
    val_loss = estimate_loss(model, val_loader.get_batch, precision,
                             config.get("eval_iters", 50))
    best_val_loss = min(best_val_loss, val_loss)
    # Record the final evaluation as a metrics.csv row too: the analyzer
    # reads metrics.csv only, so without this row every report would quote
    # the LAST INTERVAL eval instead of the model's true final quality.
    elapsed = time.time() - t_total0
    logger.record_metrics({
        "step": max_steps, "tokens_seen": tokens_seen,
        "train_loss": "", "val_loss": round(val_loss, 4),
        "val_ppl": round(perplexity_from_loss(val_loss), 2),
        "lr": round(scheduler.current_lr, 8),
        "step_time_s": "", "tokens_per_sec": round(tokens_seen / max(elapsed, 1e-6), 1),
        "gpu_mem_mb": round(peak_memory_mb(), 1), "grad_norm": "",
    })
    save_checkpoint(
        os.path.join(run_dir, "latest.pt"), base_gpt, optimizer, scheduler,
        precision, max_steps, tokens_seen, config, best_val_loss,
        is_best=(best_val_loss == val_loss))
    plot_loss_curves(run_dir)
    sync_run_to_drive(run_dir, full=True)

    summary = {
        "run_dir": run_dir,
        "experiment": config["experiment"],
        "attention": config["attention"],
        "position": config["position"],
        "dataset": config["dataset"],
        "size": config["size"],
        "seed": config["seed"],
        "final_val_loss": round(val_loss, 4),
        "best_val_loss": round(best_val_loss, 4),
        "val_ppl": round(perplexity_from_loss(best_val_loss), 2),
        "total_time_s": round(time.time() - t_total0, 1),
        "tokens_seen": tokens_seen,
        "tokens_per_sec": round(tokens_seen / max(time.time() - t_total0, 1e-6), 1),
        "peak_gpu_mem_mb": round(peak_memory_mb(), 1),
        "num_params": base_gpt.get_num_params(non_embedding=False),
        "devices_used": [int(d) for d in device_ids] if device_ids else ["cpu"],
    }
    logger.log(f"DONE | best val loss {summary['best_val_loss']} "
               f"| ppl {summary['val_ppl']} | {summary['total_time_s']}s")
    import json as _json
    with open(os.path.join(run_dir, "summary.json"), "w", encoding="utf-8") as f:
        _json.dump(summary, f, indent=2)
    return summary
