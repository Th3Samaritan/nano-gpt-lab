"""Token and positional embeddings (GPT-2 style).

WHAT
----
TokenEmbedding          : vocab_size x n_embd table, row i is the learned
                          vector of token i.
LearnedPositionEmbedding: block_size x n_embd table, row p is a LEARNED
                          vector added to the token vector at position p.
                          This is GPT-2/GPT-3's original scheme:
                              x = wte(tokens) + wpe(positions)

WHY
----
Learned absolute positions are the BASELINE of the ablation matrix
(experiment A and C). RoPE (model/rope.py) replaces this module entirely in
experiments B and D - the two approaches answer the same question
("where am I in the sequence?") in different ways:

    learned: position p has a free parameter vector; the model must learn
             from data what "distance" means, and it only has learned rows
             for p < block_size.
    RoPE   : position is encoded by rotating q/k pairs geometrically;
             relative position becomes a closed-form function (see rope.py),
             and longer sequences extrapolate far better.

HOW
----
Both are plain nn.Embedding tables so the initialization story stays in one
place (model/gpt.py applies the normal(0, 0.02) init to all weights).
"""
from __future__ import annotations

import torch
import torch.nn as nn


class TokenEmbedding(nn.Embedding):
    """Lookup table: token id -> dense vector (vocab_size x n_embd)."""

    def __init__(self, vocab_size: int, n_embd: int):
        super().__init__(vocab_size, n_embd)


class LearnedPositionEmbedding(nn.Embedding):
    """Lookup table: absolute position -> dense vector (block_size x n_embd).

    Used ONLY when config.position == "learned". When config.position ==
    "rope" the GPT module skips this table entirely so the parameter counts
    of the two schemes stay comparable and the ablation stays clean.
    """

    def __init__(self, block_size: int, n_embd: int):
        super().__init__(block_size, n_embd)
