"""Checkpoint save/load with full resume state.

WHAT
----
Every checkpoint is a single .pt dict containing (plan section 26):
    model_state_dict, optimizer_state_dict, scheduler_state_dict,
    scaler_state_dict, step, tokens_seen, config, rng_state, best_val_loss
Two files are maintained per run: latest.pt (resume) and best.pt
(lowest validation loss seen so far).

WHY
----
Colab and Kaggle sessions die arbitrarily. A resume that restores model
weights but not optimizer momentum, the LR schedule, the loss scaler or the
RNG state produces a different trajectory than an uninterrupted run - which
would silently contaminate the controlled ablation. Hence EVERYTHING that
affects the trajectory is checkpointed.

WHERE
-----
Written by training/trainer.py at checkpoint_interval steps and at the end
of training; read by trainer.py --resume, scripts/evaluate.py and
scripts/generate.py.
"""
from __future__ import annotations

import os
from typing import Any, Optional

import torch

from src.utils.seed import get_rng_state, set_rng_state


def save_checkpoint(
    path: str,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
    scaler: Any,
    step: int,
    tokens_seen: int,
    config: dict,
    best_val_loss: Optional[float],
    is_best: bool = False,
) -> None:
    """Write one .pt checkpoint; atomically replace best.pt when flagged."""
    state = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "scaler_state_dict": scaler.state_dict() if scaler is not None else None,
        "step": step,
        "tokens_seen": tokens_seen,
        "config": config,
        "rng_state": get_rng_state(),
        "best_val_loss": best_val_loss,
    }
    os.makedirs(os.path.dirname(path), exist_ok=True)
    # Write to a temp name first, then rename: a crash mid-write must never
    # leave a truncated latest.pt that looks valid.
    tmp = path + ".tmp"
    torch.save(state, tmp)
    os.replace(tmp, path)
    if is_best:
        best_path = os.path.join(os.path.dirname(path), "best.pt")
        torch.save(state, best_path)


def load_checkpoint(
    path: str,
    model: torch.nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: Any = None,
    scaler: Any = None,
) -> dict:
    """Restore a checkpoint into live objects; returns the raw state dict.

    The optimizer/scheduler/scaler args may be None (evaluation or generation
    only needs weights); their parts of the state are skipped in that case.
    """
    state = torch.load(path, map_location="cpu", weights_only=False)
    model.load_state_dict(state["model_state_dict"])
    if optimizer is not None and state.get("optimizer_state_dict") is not None:
        optimizer.load_state_dict(state["optimizer_state_dict"])
    if scheduler is not None and state.get("scheduler_state_dict") is not None:
        scheduler.load_state_dict(state["scheduler_state_dict"])
    if scaler is not None and state.get("scaler_state_dict") is not None:
        scaler.load_state_dict(state["scaler_state_dict"])
    if state.get("rng_state") is not None:
        set_rng_state(state["rng_state"])
    return state


def find_latest_checkpoint(run_dir: str) -> Optional[str]:
    """Path to latest.pt in a run dir, or None when the run never trained."""
    path = os.path.join(run_dir, "latest.pt")
    return path if os.path.exists(path) else None
