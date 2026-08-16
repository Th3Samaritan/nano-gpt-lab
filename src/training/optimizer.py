"""AdamW construction with the GPT-2/GPT-3 weight-decay convention.

WHAT
----
create_optimizer() splits parameters into two groups:
    decay group    : all 2D weights (Linear/Embedding matrices)
    no-decay group : all 1D parameters (LayerNorm gains/biases)
AdamW is applied to both groups, with weight_decay=0 for the no-decay
group. Betas (0.9, 0.95) and epsilon 1e-8 follow GPT-3's recipe.

WHY
----
Decaying LayerNorm scale/bias is pointless (they are per-feature scalars,
not interactions) and can fight the normalization itself. This split is the
standard GPT recipe and - like everything else in the recipe - is applied
IDENTICALLY to experiments A/B/C/D so it cannot explain any difference.

WHERE
-----
Called once per run from training/trainer.py; the optimizer object (with
its momentum state) is checkpointed so resume is exact.
"""
from __future__ import annotations

import inspect
from typing import List

import torch
import torch.nn as nn


def _split_decay(model: nn.Module, weight_decay: float) -> List[dict]:
    """Group named parameters by 'decay' vs 'no-decay'."""
    decay, no_decay = [], []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        # ndim >= 2 selects weight matrices; everything else (biases,
        # LayerNorm scales) goes to the no-decay group.
        if param.ndim >= 2:
            decay.append(param)
        else:
            no_decay.append(param)
    return [
        {"params": decay, "weight_decay": weight_decay},
        {"params": no_decay, "weight_decay": 0.0},
    ]


def create_optimizer(
    model: nn.Module,
    weight_decay: float,
    learning_rate: float,
    betas: tuple,
    device: str,
) -> torch.optim.Optimizer:
    """AdamW over the decay/no-decay split (see module docstring).

    fused=True uses torch's fused AdamW CUDA kernel when available (~20%
    faster step); it silently falls back on CPU, so it is safe everywhere.
    """
    fused_available = (
        "fused" in inspect.signature(torch.optim.AdamW).parameters
        and device == "cuda"
    )
    groups = _split_decay(model, weight_decay)
    return torch.optim.AdamW(
        groups, lr=learning_rate, betas=betas, eps=1e-8,
        fused=fused_available,
    )
