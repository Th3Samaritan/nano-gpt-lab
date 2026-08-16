"""MLP block (GPT-2 style) and LayerNorm wrapper.

WHAT
----
MLP: the "thinking" half of each transformer block. It expands the residual
stream to 4x width, applies GELU, and projects back:
    x -> Linear(n_embd -> 4*n_embd) -> GELU -> Linear(4*n_embd -> n_embd)
LayerNorm: normalizes each token's vector to mean 0 / variance 1 over the
embedding dimension, then rescales with learned gamma/beta.

WHY
----
GPT-2/GPT-3 use exactly this pair, so the lab's baselines must too - the
ablation matrix only varies attention + position, never the MLP.
The normalization.py file also exists because Phase 9 (plan section 35)
will swap LayerNorm -> RMSNorm here WITHOUT touching anything else - the
one-change-at-a-time rule is enforced by file boundaries.

WHERE
-----
MLP and LayerNorm are composed inside model/transformer_block.py's Block.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class MLP(nn.Module):
    """GPT-2 feed-forward network: 4x expansion, GELU, projection."""

    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias)
        # GELU (Gaussian Error Linear Unit): smooth RELU alternative. GPT-2
        # used the tanh approximation; exact GELU is used here because it is
        # the standard torch primitive and differences are negligible.
        self.gelu = nn.GELU()
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.c_proj(self.gelu(self.c_fc(x))))
