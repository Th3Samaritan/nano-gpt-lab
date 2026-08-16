"""Mixed-precision context: autocast + GradScaler management.

WHAT
----
PrecisionContext wraps the three precision regimes behind one interface:
    bf16  (Ampere+)  : autocast only - bf16 keeps fp32's exponent range so
                       no loss scaling is needed
    fp16  (T4/P100)  : autocast + GradScaler - fp16's small range requires
                       scaling the loss so tiny gradients do not underflow
                       to zero, then unscaling before the optimizer step
    fp32  (CPU)      : no autocast, no scaler - exact math for tests/smoke

WHY
----
Mixed precision roughly doubles T4 throughput and halves memory - and the
plan (section 16) demands the SAME precision mode across experiments, with
the mode recorded in config. That recording + this single implementation
point guarantee no experiment silently trains in a different precision.

WHERE
-----
One instance per training run, created by training/trainer.py, checkpointed
via its state_dict (scaler state matters for exact resume in fp16).
"""
from __future__ import annotations

from typing import Optional

import torch


class PrecisionContext:
    """Autocast + GradScaler facade (see module docstring)."""

    def __init__(self, device: str, dtype: Optional[torch.dtype]):
        self.device = device
        self.dtype = dtype
        self.use_amp = device == "cuda" and dtype is not None
        self.scaler = (
            torch.amp.GradScaler("cuda", enabled=(dtype == torch.float16))
            if self.use_amp else None
        )
        # autocast context used around forward passes
        if self.use_amp:
            self.autocast_ctx = torch.amp.autocast("cuda", dtype=dtype)
        else:
            from contextlib import nullcontext
            self.autocast_ctx = nullcontext()

    def backward(self, loss: torch.Tensor) -> None:
        """Backward through the scaler (identity when not in fp16)."""
        if self.scaler is not None:
            self.scaler.scale(loss).backward()
        else:
            loss.backward()

    def optimizer_step(self, optimizer, grad_clip: float) -> float:
        """Clip, step, update scaler. Returns the (unscaled) grad norm.

        Grad clipping MUST see the unscaled gradients: clipping scaled
        gradients against the same threshold would allow 2^scale larger
        true gradients through, which would change training dynamics.
        """
        if self.scaler is not None:
            self.scaler.unscale_(optimizer)
        grad_norm = torch.nn.utils.clip_grad_norm_(
            _params(optimizer), grad_clip) if grad_clip > 0 else 0.0
        if self.scaler is not None:
            self.scaler.step(optimizer)
            self.scaler.update()
        else:
            optimizer.step()
        return float(grad_norm) if torch.is_tensor(grad_norm) else grad_norm

    def state_dict(self) -> Optional[dict]:
        return self.scaler.state_dict() if self.scaler is not None else None

    def load_state_dict(self, state) -> None:
        if self.scaler is not None and state is not None:
            self.scaler.load_state_dict(state)


def _params(optimizer):
    """All trainable params of an optimizer's param groups (for clipping)."""
    for group in optimizer.param_groups:
        for p in group["params"]:
            if p.grad is not None:
                yield p
