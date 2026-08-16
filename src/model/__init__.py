"""Model package: the transformer, component by component."""
from src.model.attention import CausalSelfAttention, attention_vanilla, causal_mask
from src.model.embeddings import LearnedPositionEmbedding, TokenEmbedding
from src.model.flash_attention import (
    flash_attention_fused,
    flash_attention_tiled,
)
from src.model.gpt import GPT, GPTConfig
from src.model.mlp import MLP
from src.model.normalization import LayerNorm
from src.model.rope import apply_rope, build_rope_tables, rotate_half
from src.model.transformer_block import Block

__all__ = [
    "CausalSelfAttention", "attention_vanilla", "causal_mask",
    "LearnedPositionEmbedding", "TokenEmbedding",
    "flash_attention_fused", "flash_attention_tiled",
    "GPT", "GPTConfig",
    "MLP", "LayerNorm",
    "apply_rope", "build_rope_tables", "rotate_half",
    "Block",
]
