"""Optimizer study (ABLATION_PLAN section 6/7/8): Adam vs AdamW, plus
first-principles reference implementations validated against PyTorch.

WHAT
----
create_optimizer() builds the study-selected optimizer over the standard
decay/no-decay parameter split (2D weight matrices decay; LayerNorm
scalars do not).

    optimizer = "adamw" (C2, canonical baseline): DECOUPLED weight decay -
        step:  theta -= lr * (m_hat / (sqrt(v_hat) + eps) + wd * theta)
        The decay term acts directly on the weights, independently of the
        gradient moments.

    optimizer = "adam" (C1): the classic Kingma & Ba formulation with
        L2-style decay COUPLED into the gradient:
        step:  theta -= lr * (m_hat / (sqrt(v_hat) + eps)), g = grad + wd*theta
        Because the decayed gradient flows through the moment estimates,
        regularization gets modulated by the adaptive rates - the exact
        coupling AdamW was invented to remove. C1 vs C2 measures whether
        that coupling matters at our budget.

AdamReference / AdamWReference implement the update equations EXPLICITLY
(no library optimizer) for understanding (ABLATION_PLAN section 8: the
production path stays PyTorch-native; the point is that the math is
written down and tested).

WHY
----
The optimizer comparison must not become "well-tuned vs badly tuned": both
share the SAME lr, betas, eps, decay split, scheduler and clip. Only the
decay mechanism differs (section 7 primary comparison).

WHERE
-----
create_optimizer is called by training/trainer.py; the reference classes
are exercised by tests/test_optimizers.py against torch's own Adam/AdamW.
"""
from __future__ import annotations

import inspect
import math
from typing import List

import torch
import torch.nn as nn


# --------------------------------------------------------------------------
# Parameter split (decay / no-decay) - identical for Adam and AdamW
# --------------------------------------------------------------------------
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
    optimizer: str = "adamw",
) -> torch.optim.Optimizer:
    """Build the study-selected optimizer (C1 Adam / C2 AdamW).

    torch.optim.Adam's weight_decay implements the CLASSIC L2-style decay
    (added to the gradient), while torch.optim.AdamW implements decoupled
    decay - precisely the pair the study compares.
    """
    fused_available = (
        "fused" in inspect.signature(torch.optim.AdamW).parameters
        and device == "cuda"
    )
    groups = _split_decay(model, weight_decay)
    if optimizer == "adam":
        return torch.optim.Adam(groups, lr=learning_rate, betas=betas, eps=1e-8)
    if optimizer == "adamw":
        return torch.optim.AdamW(
            groups, lr=learning_rate, betas=betas, eps=1e-8,
            fused=fused_available)
    raise ValueError(f"unknown optimizer '{optimizer}' (pick adam | adamw)")


# --------------------------------------------------------------------------
# First-principles reference implementations (ABLATION_PLAN section 8)
# --------------------------------------------------------------------------
class _AdamBase:
    """The shared update math, written out explicitly.

    State per parameter tensor: m (first moment), v (second moment), t.
    Every line below is the textbook equation; nothing is hidden."""

    def __init__(self, params, lr: float, betas: tuple, eps: float,
                 weight_decay: float, decoupled: bool):
        self.params = list(params)
        self.lr, self.betas, self.eps = lr, betas, eps
        self.weight_decay = weight_decay
        self.decoupled = decoupled
        self.state = {}
        for p in self.params:
            self.state[p] = {
                "m": torch.zeros_like(p),
                "v": torch.zeros_like(p),
                "t": 0,
            }

    def step(self) -> None:
        """One update across all params:
            1. g = grad (+ wd * theta if decay is COUPLED, i.e. Adam-L2)
            2. m = beta1*m + (1-beta1)*g
            3. v = beta2*v + (1-beta2)*g^2
            4. bias-corrected: m_hat = m/(1-beta1^t), v_hat = v/(1-beta2^t)
            5. theta -= lr * m_hat/(sqrt(v_hat)+eps)  (+ lr*wd*theta if
               decay is DECOUPLED, i.e. AdamW)
        """
        b1, b2 = self.betas
        with torch.no_grad():
            for p in self.params:
                if p.grad is None:
                    continue
                st = self.state[p]
                st["t"] += 1
                g = p.grad
                if self.weight_decay and not self.decoupled:
                    # Adam (classic): L2 decay rides the gradient, so the
                    # moments see it and the adaptive rates modulate it.
                    g = g + self.weight_decay * p
                st["m"].mul_(b1).add_(g, alpha=1 - b1)
                st["v"].mul_(b2).addcmul_(g, g, value=1 - b2)
                m_hat = st["m"] / (1 - b1 ** st["t"])
                v_hat = st["v"] / (1 - b2 ** st["t"])
                update = m_hat / (v_hat.sqrt() + self.eps)
                if self.weight_decay and self.decoupled:
                    # AdamW: decay is decoupled - a plain weight shrink
                    # applied every step regardless of gradients/moments.
                    p.mul_(1 - self.lr * self.weight_decay)
                p.add_(update, alpha=-self.lr)


class AdamReference(_AdamBase):
    """Reference Adam with classic (coupled) L2 weight decay."""

    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8,
                 weight_decay=0.0):
        super().__init__(params, lr, betas, eps, weight_decay,
                         decoupled=False)


class AdamWReference(_AdamBase):
    """Reference AdamW with decoupled weight decay."""

    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8,
                 weight_decay=0.0):
        super().__init__(params, lr, betas, eps, weight_decay,
                         decoupled=True)


def optimizer_state_memory(model: nn.Module, optimizer_name: str) -> float:
    """Optimizer-state memory in MB (section 14 memory accounting).

    Adam-family optimizers keep two fp32 moments per trainable parameter:
    state_bytes = 2 * 4 bytes * num_trainable_params."""
    n = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return 2 * 4 * n / (1024 * 1024)
