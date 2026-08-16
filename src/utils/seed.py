"""Deterministic seeding + RNG state capture for checkpointing.

WHAT
----
seed_all(seed) seeds python's random, numpy, torch CPU and torch CUDA.
get_rng_state() / set_rng_state() snapshot and restore all of them so a
resumed run continues exactly as if it had never stopped.

WHY
----
The plan's ablation matrix demands that A/B/C/D differ in ONE architectural
variable. If two runs also differed in RNG state they would differ in data
order too, and loss differences could not be attributed to the architecture.
The fixed seed set {42, 123, 2026} is applied identically to every variant.

WHERE
-----
Called once at the start of scripts/train.py, and stored inside every
checkpoint (plan section 26 lists rng_state as a required checkpoint field).
"""
from __future__ import annotations

import random

import numpy as np
import torch


def seed_all(seed: int) -> None:
    """Seed every RNG the training pipeline touches, once, at run start.

    Order matters conceptually but not practically; the key point is that all
    consumers (data shuffling via numpy, dropout via torch, python sampling
    for anything misc) share the same seed sequence across experiments.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    # The next two flags make CUDA kernels deterministic where possible.
    # They cost some throughput; they are ON because the lab's primary goal
    # is comparability, not raw speed (plan section 7).
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_rng_state() -> dict:
    """Snapshot every RNG so a checkpoint can restore exact data order.

    torch.cuda RNG exists even on CPU-only machines as a state object, so it
    is captured unconditionally and restored only if CUDA is available.
    """
    state = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "torch_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }
    return state


def set_rng_state(state: dict) -> None:
    """Restore a snapshot produced by get_rng_state (resume support)."""
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if state.get("torch_cuda") is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["torch_cuda"])
