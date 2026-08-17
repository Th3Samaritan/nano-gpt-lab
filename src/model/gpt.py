"""GPT model assembly: config, weight init, forward, loss, generation.

WHAT
----
GPTConfig: every architectural knob in one dataclass. Two of its fields are
the experimental variables of the whole lab:
    attention : "vanilla" | "flash_tiled" | "flash_fused"
    position  : "learned" | "rope" | "none"
Everything else (depth, width, heads, context, dropout, bias) is held
constant across the A/B/C/D ablation.

GPT: the decoder-only stack
    wte (token table) [+ wpe (position table) if position == "learned"]
    -> n_layer x Block
    -> final LayerNorm
    -> lm_head (the SAME matrix as wte - weight tying, like GPT-2)

WHY
----
Weight tying halves the vocabulary-embedding parameter cost and was used by
GPT-2/GPT-3. The residual-projection init scaling (0.02/sqrt(2*n_layer)) is
the GPT-2 scheme that keeps the variance of the residual stream ~constant
with depth - changing init between variants would contaminate the ablation,
so it lives here once, for all experiments.

WHERE
-----
Instantiated by scripts/train.py and tests; training loop in
src/training/trainer.py; generation in src/evaluation/generation.py.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.model.embeddings import LearnedPositionEmbedding, TokenEmbedding
from src.model.normalization import LayerNorm
from src.model.rope import build_rope_tables
from src.model.transformer_block import Block


@dataclass
class GPTConfig:
    """All architectural knobs. Defaults = the 'nano' research model."""

    block_size: int = 256        # context length (max tokens per sequence)
    vocab_size: int = 4096       # BPE vocabulary size
    n_layer: int = 4             # transformer blocks (depth)
    n_head: int = 4              # attention heads per block
    n_embd: int = 256            # embedding / hidden dimension
    dropout: float = 0.0         # GPT-2 pretraining used 0.0; kept 0.0
    bias: bool = True            # Linear biases (GPT-2 had them)

    # --- the two experimental variables ----------------------------------
    attention: str = "vanilla"   # vanilla | flash_tiled | flash_fused
    position: str = "learned"    # learned | rope | none

    # --- Study B variable: the FFN activation ----------------------------
    activation: str = "gelu"     # gelu | relu | silu | swiglu
    swiglu_matched: bool = True  # True: hidden width matched to GELU param
                                 # count (fairness rule, ABLATION_PLAN sec 5)

    # --- RoPE / tiled-flash details ------------------------------------
    rope_base: float = 10000.0   # RoFormer's theta base
    flash_tile_size: int = 64    # tile size of the first-principles tiler

    def __post_init__(self):
        assert self.n_embd % self.n_head == 0, "n_embd must be divisible by n_head"
        assert self.attention in ("vanilla", "flash_tiled", "flash_fused")
        assert self.position in ("learned", "rope", "none")
        assert self.activation in ("gelu", "relu", "silu", "swiglu")

    @property
    def head_dim(self) -> int:
        return self.n_embd // self.n_head


class GPT(nn.Module):
    """Decoder-only transformer: GPT-2 architecture, ablatable attention."""

    def __init__(self, config: GPTConfig):
        super().__init__()
        self.config = config

        # Token table. When position == "learned" a SECOND table is added -
        # its (block_size x n_embd) parameters are the only architectural
        # parameter-count difference between the two position schemes.
        self.wte = TokenEmbedding(config.vocab_size, config.n_embd)
        self.wpe = (
            LearnedPositionEmbedding(config.block_size, config.n_embd)
            if config.position == "learned" else None
        )
        self.drop = nn.Dropout(config.dropout)

        # The repeating block stack (see transformer_block.py).
        self.h = nn.ModuleList([Block(config) for _ in range(config.n_layer)])
        self.ln_f = LayerNorm(config.n_embd, bias=config.bias)
        # lm_head is NOT a module: weight tying shares wte.weight (GPT-2).
        self.lm_head_weight = self.wte.weight

        # RoPE tables are built lazily on first forward for the right
        # device/dtype; cached per (device, dtype) so they are built once.
        self._rope_cache: dict = {}

        self.apply(self._init_weights)
        self._init_residual_projections(config.n_layer)

    # ------------------------------------------------------------------ #
    # Weight initialization (GPT-2 scheme - frozen across all variants)  #
    # ------------------------------------------------------------------ #
    def _init_weights(self, module: nn.Module) -> None:
        """normal(0, 0.02) for weights, zeros for biases (GPT-2)."""
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def _init_residual_projections(self, n_layer: int) -> None:
        """Scale down projections that feed a residual ADD.

        WHY: a block adds its output to the stream, so un-scaling its
        projection would grow stream variance by O(n_layer). Scaling the
        two residual projections (attention c_proj and MLP c_proj) by
        1/sqrt(2*n_layer) keeps the variance stable at depth - the GPT-2
        recipe, applied identically to every experiment.
        """
        for name, param in self.named_parameters():
            if name.endswith("c_proj.weight"):
                torch.nn.init.normal_(
                    param, mean=0.0, std=0.02 / math.sqrt(2 * n_layer))

    # ------------------------------------------------------------------ #
    # RoPE tables                                                        #
    # ------------------------------------------------------------------ #
    def _rope_tables(self, T: int, device, dtype):
        """cos/sin of shape (T, head_dim) for positions 0..T-1, cached."""
        key = (T, str(device), str(dtype))
        if key not in self._rope_cache:
            cos, sin = build_rope_tables(
                self.config.head_dim, T, self.config.rope_base, device, dtype)
            self._rope_cache = {key: (cos, sin)}  # keep only latest (memory)
        return self._rope_cache[key]

    # ------------------------------------------------------------------ #
    # Forward                                                            #
    # ------------------------------------------------------------------ #
    def forward(self, idx: torch.Tensor, targets: Optional[torch.Tensor] = None):
        """idx: (B, T) token ids. Returns (logits, loss) - loss None if no targets."""
        B, T = idx.shape
        device = idx.device
        assert T <= self.config.block_size, \
            f"sequence {T} longer than block_size {self.config.block_size}"

        # Token embedding + optional learned position table (experiment
        # variable #2). With RoPE there is NO table: positions enter only
        # inside attention via rotation.
        tok_emb = self.wte(idx)                       # (B, T, n_embd)
        if self.wpe is not None:
            pos = torch.arange(0, T, device=device).unsqueeze(0)  # (1, T)
            tok_emb = tok_emb + self.wpe(pos)

        # RoPE tables + causal mask built ONCE and shared by all blocks.
        cos = sin = None
        if self.config.position == "rope":
            cos, sin = self._rope_tables(T, device, tok_emb.dtype)
        mask = None
        if self.config.attention != "flash_fused":
            # fused backend uses is_causal=True internally (no mask tensor)
            from src.model.attention import causal_mask
            mask = causal_mask(T, device)

        x = self.drop(tok_emb)
        for block in self.h:
            x = block(x, cos, sin, mask)
        x = self.ln_f(x)

        # Weight tying: logits = x @ wte.weight^T
        logits = F.linear(x, self.lm_head_weight)     # (B, T, vocab)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss

    # ------------------------------------------------------------------ #
    # Generation                                                         #
    # ------------------------------------------------------------------ #
    @torch.no_grad()
    def generate(self, idx: torch.Tensor, max_new_tokens: int,
                 temperature: float = 0.8, top_k: int | None = None,
                 top_p: float | None = None) -> torch.Tensor:
        """Autoregressive sampling: predict next token, append, repeat.

        temperature: <1 sharpens, >1 flattens the distribution.
        top_k: keep only the k most probable tokens (GPT-2's sampler).
        top_p: nucleus sampling - keep the smallest set holding p mass.
        Both None = plain temperature sampling.
        """
        for _ in range(max_new_tokens):
            # Crop to the last block_size tokens: the model's window slides.
            idx_cond = idx if idx.size(1) <= self.config.block_size \
                else idx[:, -self.config.block_size:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / max(temperature, 1e-8)  # (B, vocab)

            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = float("-inf")
            if top_p is not None:
                sorted_logits, sorted_idx = torch.sort(logits, descending=True)
                cum = torch.softmax(sorted_logits, dim=-1).cumsum(dim=-1)
                remove = cum > top_p
                remove[..., 1:] = remove[..., :-1].clone()
                remove[..., 0] = False
                logits = logits.scatter(
                    1, sorted_idx, sorted_logits.masked_fill(remove, float("-inf")))

            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)  # (B, 1)
            idx = torch.cat((idx, next_token), dim=1)
        return idx

    # ------------------------------------------------------------------ #
    # Bookkeeping                                                        #
    # ------------------------------------------------------------------ #
    def get_num_params(self, non_embedding: bool = True) -> int:
        """Parameter count. non_embedding=True excludes wte/wpe from the
        total used for compute comparisons (GPT-3's convention) but both
        numbers are logged in model_summary.txt."""
        n = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n -= self.wte.weight.numel()
            if self.wpe is not None:
                n -= self.wpe.weight.numel()
        return n

    def param_breakdown(self) -> dict:
        """Parameter accounting per ABLATION_PLAN section 15:
        total / trainable / embedding / attention / FFN / LM-head / other.

        The LM head is weight-tied (wte), so it costs 0 extra parameters -
        reported explicitly because section 15 demands every experiment
        state its parameter story."""
        counts = {"total": 0, "trainable": 0, "embedding": 0,
                  "attention": 0, "ffn": 0, "lm_head": 0, "other": 0}

        def add(module: nn.Module, bucket: str) -> None:
            for p in module.parameters():
                n = p.numel()
                counts[bucket] += n
                counts["total"] += n
                if p.requires_grad:
                    counts["trainable"] += n

        add(self.wte, "embedding")
        if self.wpe is not None:
            add(self.wpe, "embedding")
        for block in self.h:
            add(block.attn, "attention")
            add(block.mlp, "ffn")
            for ln in (block.ln_1, block.ln_2):
                add(ln, "other")
        add(self.ln_f, "other")
        # lm_head is wte itself: tied, zero extra parameters
        return counts

    def flops_per_token(self) -> float:
        """Approx FLOPS per token: forward 2N, backward 4N -> 6N params."""
        return 6 * self.get_num_params(non_embedding=False)

    def model_summary(self) -> str:
        """Human-readable architecture summary saved into every run dir."""
        cfg = self.config
        breakdown = self.param_breakdown()
        return "\n".join([
            f"model           : GPT decoder-only",
            f"layers          : {cfg.n_layer}",
            f"heads           : {cfg.n_head}",
            f"embedding dim   : {cfg.n_embd}",
            f"context length  : {cfg.block_size}",
            f"vocab size      : {cfg.vocab_size}",
            f"attention       : {cfg.attention}",
            f"position        : {cfg.position}",
            f"activation      : {cfg.activation}"
            + (" (param-matched)" if cfg.activation == "swiglu"
               and cfg.swiglu_matched else
               " (architecture-native)" if cfg.activation == "swiglu" else ""),
            f"dropout         : {cfg.dropout}",
            f"bias            : {cfg.bias}",
            f"total params    : {breakdown['total']:,}",
            f"trainable       : {breakdown['trainable']:,}",
            f"embedding       : {breakdown['embedding']:,}",
            f"attention       : {breakdown['attention']:,}",
            f"FFN             : {breakdown['ffn']:,}",
            f"LM head (tied)  : {breakdown['lm_head']:,} (shared with wte)",
            f"norm/other      : {breakdown['other']:,}",
            f"flops/token     : ~{self.flops_per_token()/1e6:.1f} MFLOPs",
        ])
