"""Normalization wrappers.

WHAT
----
LayerNorm: the pre-LN normalization used in every GPT-2 block
    y = (x - mean(x)) / sqrt(var(x) + eps) * gamma + beta
where mean/var run over the LAST dimension (per token, per feature vector).

WHY
----
Stabilizes training by keeping activation scale roughly constant before
attention and before the MLP. GPT-2 normalizes BEFORE each sublayer
("pre-LN"), which is why no normalization appears after the residuals in
model/transformer_block.py.

Phase 9 (plan section 35) will add RMSNorm here as the one-at-a-time
LayerNorm replacement; keeping this file tiny makes that diff reviewable.
"""
from __future__ import annotations

import torch.nn as nn


class LayerNorm(nn.Module):
    """Standard LayerNorm with optional learned affine parameters."""

    def __init__(self, n_embd: int, bias: bool = True):
        super().__init__()
        self.ln = nn.LayerNorm(n_embd, bias=bias)
        # torch's LayerNorm normalizes over the last dim by default, which
        # is exactly the per-token normalization GPT-2 wants.

    def forward(self, x):
        return self.ln(x)
