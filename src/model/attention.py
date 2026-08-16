"""Vanilla causal self-attention written as the bare mathematics.

WHAT
----
attention_vanilla() is the "naked" implementation the plan demands
(section 30) - every mathematical operation visible, nothing fused:

    Q = X W_q          (projections, done by CausalSelfAttention below)
    K = X W_k
    V = X W_v
    S = Q K^T / sqrt(d_head)        <- attention logits (B, nh, T, T)
    S = S.masked_fill(causal, -inf) <- block future tokens
    A = softmax(S, dim=-1)          <- attention probabilities
    Y = A V

This materializes the full T x T score matrix - O(T^2) memory, which is
EXACTLY what FlashAttention exists to avoid (flash_attention.py). Keeping
this naive version as the reference is the whole point: every optimized
path is validated against it in tests/test_attention.py.

CausalSelfAttention is the multi-head module: a single Linear produces
Q, K, V concatenated (GPT-2 style), heads are split/transposed, RoPE is
applied when config.position == "rope", and the selected backend computes
the output. The backend is chosen by config.attention:

    "vanilla"     -> attention_vanilla (this module)
    "flash_tiled" -> flash_attention.flash_attention_tiled
    "flash_fused" -> flash_attention.flash_attention_fused

WHY
---
A/B/C/D must differ ONLY in attention backend + position scheme, so one
module owns the dispatch and all backends share identical QKV handling,
init, and projection - guaranteeing any measured difference comes from the
backend itself, not from wiring drift.

WHERE
-----
One instance per transformer block (model/transformer_block.py); assembled
into the full GPT in model/gpt.py.
"""
from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.model.rope import apply_rope


def causal_mask(T: int, device: torch.device | str) -> torch.Tensor:
    """(1, 1, T, T) boolean mask: True = allowed, False = blocked.

    Causal attention lets position i attend to positions <= i only, so the
    mask is the lower triangle of a T x T matrix (including the diagonal).
    """
    return torch.tril(torch.ones(T, T, device=device, dtype=torch.bool))[
        None, None, :, :
    ]


def attention_vanilla(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """The naked attention math (see module docstring).

    q, k, v: (B, nh, T, hd). mask: boolean (1,1,T,T) or None (non-causal).
    -inf fill is exact (not a large negative number): softmax turns -inf
    entries into exactly 0.0, so future tokens contribute NOTHING - the
    no-future-leakage property verified in tests/test_attention.py.
    """
    scale = q.shape[-1] ** -0.5                    # 1/sqrt(d_head)
    scores = q @ k.transpose(-2, -1) * scale       # S = QK^T / sqrt(d)
    if mask is not None:
        scores = scores.masked_fill(~mask, float("-inf"))
    att = F.softmax(scores, dim=-1)                # A = softmax(S)
    return att @ v                                 # Y = A V


class CausalSelfAttention(nn.Module):
    """Multi-head causal attention with pluggable backend + RoPE option."""

    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0, \
            "n_embd must be divisible by n_head"
        self.n_embd = config.n_embd
        self.n_head = config.n_head
        self.head_dim = config.n_embd // config.n_head
        self.attention = config.attention
        self.position = config.position
        self.dropout = config.dropout
        # One Linear for Q, K and V stacked: weight is [3*n_embd, n_embd],
        # the first third is W_q, then W_k, then W_v - exactly GPT-2's c_attn.
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)
        # Output projection mixes the heads back into the residual stream.
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)

    def forward(
        self,
        x: torch.Tensor,
        cos: Optional[torch.Tensor] = None,
        sin: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # x: (B, T, n_embd) -> q, k, v each (B, T, n_embd)
        B, T, _ = x.shape
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
        # Reshape to (B, nh, T, hd): heads become an independent dimension
        # so every head computes attention over the same positions.
        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)

        # RoPE rotates q and k by absolute position BEFORE the dot product;
        # the resulting scores encode relative position (see rope.py).
        # cos/sin are (T_table, head_dim): slice ROWS by position ([:T]),
        # never columns - the row index is the position.
        if self.position == "rope":
            q, k = apply_rope(q, k, cos[:T], sin[:T])

        # Backend dispatch - the only architectural variable of this module.
        if self.attention == "vanilla":
            y = attention_vanilla(q, k, v, mask)
        elif self.attention == "flash_tiled":
            from src.model.flash_attention import flash_attention_tiled
            y = flash_attention_tiled(q, k, v, mask)
        elif self.attention == "flash_fused":
            from src.model.flash_attention import flash_attention_fused
            y = flash_attention_fused(q, k, v)
        else:
            raise ValueError(f"unknown attention backend: {self.attention}")

        # (B, nh, T, hd) -> (B, T, n_embd), then the output projection.
        y = y.transpose(1, 2).contiguous().view(B, T, self.n_embd)
        y = self.resid_dropout(self.c_proj(y))
        return y
