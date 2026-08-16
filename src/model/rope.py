"""Rotary Positional Embeddings (RoPE) from first principles.

WHAT
----
RoPE (Su et al., "RoFormer", 2021) encodes position by ROTATING pairs of
dimensions of q and k through angles proportional to the absolute position.

    For position m and dimension pair 2i, 2i+1:
        angle(m, i) = m * theta^{-2i/d}
        [x_2i ; x_2i+1] -> rotation by angle(m, i)

The magic is what happens to the attention score. Rotating q_m and k_n
through angles that depend on m and n gives:

    <R(m) q, R(n) k> = <R(m-n) q, k>

i.e. the dot product depends on the RELATIVE position (m-n) alone. The
model sees relative offsets - exactly what attention should attend to -
without ever learning an embedding table for absolute positions.

WHY
----
This is the second variable of the ablation matrix (experiments B and D).
Versus learned absolute embeddings, RoPE makes two testable claims:
    1. relative position is the right inductive bias (better val loss at
       equal tokens, plan section 18's "fewer tokens to same quality")
    2. sequences longer than training context extrapolate (the rotation
       for position m is defined for ANY m - there is no table to run out
       of)

WHERE
-----
Tables are built once per forward pass in model/gpt.py and passed to every
transformer block, which applies them to q and k inside CausalSelfAttention
BEFORE attention (rotating after projection is the standard scheme).
"""
from __future__ import annotations

from typing import Tuple

import torch


def build_rope_tables(
    head_dim: int,
    max_seq_len: int,
    base: float = 10000.0,
    device: torch.device | str = "cpu",
    dtype: torch.dtype = torch.float32,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Precompute cos/sin tables of shape (max_seq_len, head_dim).

    HOW (the construction)
    ----------------------
    1. inv_freq[i] = base^(-2i/head_dim) for i in [0, head_dim/2):
       the classic sinusoidal frequencies from the original Transformer
       paper, but used as ROTATION ANGLES instead of embedding entries.
       Low i -> slow rotation (long-range structure), high i -> fast
       rotation (local structure). Base 10000 is the RoFormer standard.
    2. angles = outer(position, inv_freq): every (m, i) pair gets its angle.
    3. Each angle is duplicated to the dimension PAIR (2i, 2i+1) and
       cos/sin are returned separately so apply_rope() can do
           x*cos + rotate_half(x)*sin
       which is the standard efficient form of a 2D rotation matrix:
           [cos -sin] [x1]   =  x1*cos - x2*sin
           [sin  cos] [x2]      x1*sin + x2*cos
    """
    i = torch.arange(0, head_dim // 2, device=device, dtype=torch.float32)
    inv_freq = 1.0 / (base ** (i / (head_dim // 2)))
    positions = torch.arange(max_seq_len, device=device, dtype=torch.float32)
    angles = torch.outer(positions, inv_freq)          # (T, head_dim/2)
    emb = torch.cat([angles, angles], dim=-1)          # pair layout (2i,2i+1)
    return emb.cos().to(dtype), emb.sin().to(dtype)


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """The 90-degree half that completes the rotation trick.

    Splits the last dim into two halves (x1, x2) and returns (-x2, x1).
    Combined with x*cos:  x1*cos - x2*sin ; x2*cos + x1*sin  = 2D rotation.
    Input x has shape (..., head_dim) - batch/heads/seq dims untouched.
    """
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2:]
    return torch.cat([-x2, x1], dim=-1)


def apply_rope(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Rotate q and k by their ABSOLUTE positions (see module docstring).

    q, k   : (B, n_head, T, head_dim) - already projected per head
    cos/sin: (T, head_dim) tables from build_rope_tables, sliced to T.
             Broadcasting applies row m of the table to every position-m
             vector across batch and heads.
    """
    q = q * cos + rotate_half(q) * sin
    k = k * cos + rotate_half(k) * sin
    return q, k
