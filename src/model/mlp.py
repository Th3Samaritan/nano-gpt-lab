"""MLP (feed-forward) block: four activations, first principles.

WHAT
----
The FFN is the "thinking" half of each transformer block:
    x -> Linear(n -> h) -> ACT -> Linear(h -> n)      (classic, GPT-2 shape)
or, for SwiGLU, a GATED variant:
    x -> Linear(n -> h) (gate, SiLU)  ⊙  Linear(n -> h) (values) -> Linear(h -> n)

Supported activations (ABLATION_PLAN.md, Study B):
    gelu    B1  - GPT-2 baseline: smooth ReLU alternative
    relu    B2  - the classic: max(x, 0)
    silu    B3  - Swish/SiLU: x * sigmoid(x) - the smooth, non-monotonic one
    swiglu  B4  - SwiGLU (Shazeer 2020): gate ⊙ values, NOT a scalar swap

WHY
----
The plan's core rule: change ONE variable. The activation is the ONLY
difference between B1-B4 configs; attention/position/optimizer/recipe are
held at the canonical baseline (vanilla + learned + AdamW).

SwiGLU fairness (ABLATION_PLAN section 5): a gated FFN has different
parameter counts than the classic one, so TWO interpretations are shipped:
    swiglu_matched=True  : hidden width h = round(8n/3) so the WEIGHT count
                           (3·n·h) matches the GELU baseline (8·n²). This
                           answers "does the GATE help, or just the width?"
    swiglu_matched=False : h = 4n, the conventional LLaMA-style width.
                           This answers "what does the architecture-native
                           SwiGLU cost/win?"

WHERE
-----
One instance per Block (model/transformer_block.py). The residual-projection
init in model/gpt.py targets names ending in 'c_proj.weight', which covers
both the classic and the gated layout.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class MLP(nn.Module):
    """Feed-forward network with a config-selected activation (Study B)."""

    def __init__(self, config):
        super().__init__()
        n = config.n_embd
        self.activation = config.activation
        self.dropout = nn.Dropout(config.dropout)

        if self.activation == "swiglu":
            # GATED FFN: y = W_proj ( SiLU(W_gate x) ⊙ W_up x ).
            # The gate decides WHICH values pass; that is structure, not a
            # scalar nonlinearity - hence the two-width fairness rule.
            # Parameter-matched: 2·n·h + h·n = 3nh weights must equal the
            # classic 8n², so h = round(8n/3) (~2.67n for large n).
            h = round(8 * n / 3) if config.swiglu_matched else 4 * n
            self.c_gate = nn.Linear(n, h, bias=config.bias)
            self.c_up = nn.Linear(n, h, bias=config.bias)
            self.c_proj = nn.Linear(h, n, bias=config.bias)
        else:
            # Classic two-layer FFN with a scalar activation.
            self.c_fc = nn.Linear(n, 4 * n, bias=config.bias)
            self.c_proj = nn.Linear(4 * n, n, bias=config.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.activation == "swiglu":
            # elementwise product of the silu'd gate and the value stream
            h = F.silu(self.c_gate(x)) * self.c_up(x)
            return self.dropout(self.c_proj(h))
        if self.activation == "relu":
            h = F.relu(self.c_fc(x))
        elif self.activation == "silu":
            h = F.silu(self.c_fc(x))
        else:  # gelu (baseline, B1)
            h = F.gelu(self.c_fc(x))
        return self.dropout(self.c_proj(h))
