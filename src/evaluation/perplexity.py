"""Perplexity conversion.

WHAT
----
perplexity = exp(loss), where loss is the average cross-entropy (natural
log base) per token. Intuition for a layman: perplexity is roughly "how
many tokens the model is effectively choosing between at each step" -
10 means a 1-in-10 guess, 100 means a 1-in-100 guess. LOWER = better.

WHY
----
Loss differences of 0.1 are hard to read; perplexity turns them into a
multiplicative ratio (exp(0.1) ~= 10% better odds per token), which reads
naturally in the comparison tables and layman-facing charts.
"""
from __future__ import annotations

import math


def perplexity_from_loss(loss: float) -> float:
    """exp(loss): the model's average per-token branching factor."""
    return math.exp(loss)
