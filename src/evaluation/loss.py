"""Unified loss estimation used by every experiment (plan section 27).

WHAT
----
estimate_loss() averages the cross-entropy over eval_iters random batches
with the model in eval mode and gradients disabled. It takes a batch
factory (train or val split) so one function serves both.

WHY
----
A single random batch is a noisy estimate; eval_iters batches average the
noise away. Using the SAME estimator for every variant means a loss
difference between A/B/C/D cannot be a difference in measurement procedure.
The model is returned to train mode afterward - a silently-left eval-mode
(model.eval() disables dropout) would quietly change training.
"""
from __future__ import annotations

from typing import Callable

import torch
import torch.nn as nn

from src.utils.device import get_device


def estimate_loss(
    model: nn.Module,
    get_batch: Callable[[], tuple],
    precision,
    eval_iters: int,
) -> float:
    """Mean loss over eval_iters batches from the given split's get_batch."""
    device = get_device()
    model.eval()
    losses = torch.zeros(eval_iters, device="cpu")
    with torch.no_grad():
        for k in range(eval_iters):
            x, y = get_batch()
            x, y = x.to(device), y.to(device)
            with precision.autocast_ctx:
                _, loss = model(x, y)
            losses[k] = loss.item() if loss is not None else float("nan")
    model.train()  # restore train mode: dropout must behave consistently
    return float(losses.mean().item())
