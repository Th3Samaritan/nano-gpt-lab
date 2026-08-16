"""The transformer block: attention + MLP with residual streams.

WHAT
----
One Block is the repeating unit of GPT-2 (pre-LN layout):

    x ──> LayerNorm ──> SelfAttention ──> (+) ──> LayerNorm ──> MLP ──> (+)
         (residual stream carried across both sublayers)

The two additions are the RESIDUAL CONNECTIONS: attention and MLP each
produce a DELTA that is added back to the stream, so information can flow
through 12-48 layers without being squeezed through the sublayers.

WHY
----
Everything here is frozen across experiments A/B/C/D. The ONLY variables
are attention backend and position scheme, both selected upstream in
model/attention.py - the ablation's one-variable-at-a-time discipline is
physically located in that file, not here.

WHERE
-----
Stacked n_layer times in model/gpt.py. cos/sin (RoPE tables) and the causal
mask are computed ONCE per forward in GPT and passed down, so 12 blocks
never recompute the same O(T) tables.
"""
from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn

from src.model.attention import CausalSelfAttention
from src.model.mlp import MLP
from src.model.normalization import LayerNorm


class Block(nn.Module):
    """One pre-LN transformer block (attention sublayer + MLP sublayer)."""

    def __init__(self, config):
        super().__init__()
        self.ln_1 = LayerNorm(config.n_embd, bias=config.bias)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = LayerNorm(config.n_embd, bias=config.bias)
        self.mlp = MLP(config)

    def forward(
        self,
        x: torch.Tensor,
        cos: Optional[torch.Tensor] = None,
        sin: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # Residual 1: attention delta added back to the stream
        x = x + self.attn(self.ln_1(x), cos, sin, mask)
        # Residual 2: MLP delta added back to the stream
        x = x + self.mlp(self.ln_2(x))
        return x
