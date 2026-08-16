"""FlashAttention - both the first-principles tiled version and the fused one.

WHAT
----
FlashAttention (Dao et al., 2022) computes EXACTLY the same softmax
attention as attention_vanilla but never materializes the T x T score
matrix. The T x T matrix is the memory bottleneck: at block 1024 with batch
32 and 12 heads it is 32*12*1024*1024 floats ~= 1.6 GB of activations that
must be kept for the backward pass. FlashAttention splits the keys into
tiles and, for each tile, computes softmax *locally* with a running
maximum, then re-normalizes the partial results (online softmax). Peak
activation memory drops to O(tile_size), enabling bigger batches/context.

WHY (first principles)
----------------------
flash_attention_tiled() implements Algorithm 1 of the paper in ~25 lines of
pure PyTorch - tile loop, running max m, running sum l, rescale factor
exp(m_old - m_new). It is the educational "what the CUDA kernel is doing"
version. It is slow in Python because of the tile loop, but it is used to
PROVE the algorithm numerically (tests/test_attention.py) and can run real
training with --attention flash_tiled.

flash_attention_fused() calls F.scaled_dot_product_attention, which on GPU
dispatches to the actual FlashAttention-2 / memory-efficient CUDA kernels,
and falls back to the math path on CPU. It is bit-exact-enough to the
vanilla reference (tolerance-tested) and is the production path for
experiments C and D.

WHERE
-----
Backend selected by config.attention = "flash_tiled" | "flash_fused" in
model/attention.py (CausalSelfAttention.forward).
"""
from __future__ import annotations

from typing import Optional

import torch
import torch.nn.functional as F

# FILL is a large negative number instead of -inf: the online softmax in the
# tiled version computes exp(s - m) where m can be -inf for fully-masked
# rows, and exp(nan) would poison the whole tile. With FILL, masked scores
# contribute exp(~-1e9) = 0, which is numerically identical to -inf for
# softmax, while keeping every intermediate finite.
FILL = -1.0e9


def flash_attention_tiled(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
    tile_size: int = 64,
) -> torch.Tensor:
    """FlashAttention algorithm 1 (online softmax), pure PyTorch.

    WHAT THE LOOP DOES (per query row, all heads/batch vectorized)
    ---------------------------------------------------------------
    For each key/value tile j:
        s_j  = q k_j^T * scale              <- tile of scores (no T x T)
        s_j  = masked scores (-inf/FILL)
        m_j  = rowmax(s_j)                  <- tile's local max
        p_j  = exp(s_j - m_j)               <- tile-local softmax numerator
        l_j  = rowsum(p_j)                  <- tile-local denominator

    Global denominator/numerator update (the core of online softmax):
        m_new = max(m_old, m_j)
        O = O * exp(m_old - m_new) + p_j v_j   <- rescale old output
        l = l * exp(m_old - m_new) + l_j       <- rescale old denominator
        m = m_new

    After all tiles:  Y = O / l.  This is EXACTLY softmax(QK^T)V with the
    softmax computed in blocks - the rescaling exp(m_old - m_new) is an
    algebraic identity, not an approximation, which is why tests demand
    tiled == vanilla to 1e-5.

    WHY THE MASK IS APPLIED PER TILE: masking inside the tile loop keeps the
    peak memory O(tile), the same reason the real kernel fuses it.
    """
    B, nh, T, hd = q.shape
    scale = hd ** -0.5
    O = torch.zeros_like(q)                       # running output (B,nh,T,hd)
    l = torch.zeros(B, nh, T, 1, device=q.device, dtype=torch.float32)
    m = torch.full((B, nh, T, 1), float("-inf"), device=q.device,
                   dtype=torch.float32)
    for j in range(0, T, tile_size):
        kj = k[:, :, j:j + tile_size]             # this tile of keys/values
        vj = v[:, :, j:j + tile_size]
        s = q @ kj.transpose(-2, -1) * scale      # (B,nh,T,tile)
        if mask is not None:
            # mask is (1,1,T,T): slice the columns belonging to this tile
            s = s.masked_fill(~mask[:, :, :, j:j + tile_size], FILL)
        m_j = s.max(dim=-1, keepdim=True).values  # tile-local row max
        m_new = torch.maximum(m, m_j)
        p = torch.exp(s - m_new)                  # stable tile softmax num.
        l_new = l * torch.exp(m - m_new) + p.sum(dim=-1, keepdim=True)
        O = O * torch.exp(m - m_new) + p @ vj     # rescale + accumulate
        m, l = m_new, l_new
    return O / l                                  # final normalization


def flash_attention_fused(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
) -> torch.Tensor:
    """PyTorch's fused attention: FlashAttention-2 / mem-efficient kernels.

    is_causal=True is passed instead of a materialized mask because the CUDA
    kernels implement causal masking natively - faster AND the reason this
    path allocates no T x T tensor anywhere.

    On CPU this falls back to the (optimized) math path, which is why all
    tests run identically on a CPU-only machine. Numerical agreement with
    attention_vanilla is asserted in tests/test_attention.py (tolerance
    accommodates the fused kernel's different summation order in fp16).
    """
    return F.scaled_dot_product_attention(
        q, k, v, is_causal=True, dropout_p=0.0
    )
