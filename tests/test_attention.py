"""Attention tests (plan section 29 + 31): shape, causality, and the
critical numerical-equivalence checks between every backend.

The equivalence tests are the load-bearing wall of this lab: if tiled/flash
did not reproduce vanilla attention to tight tolerance, every loss number
from experiments C/D would be uninterpretable.
"""
import torch

from src.model.attention import CausalSelfAttention, attention_vanilla, causal_mask
from src.model.flash_attention import flash_attention_fused, flash_attention_tiled
from src.model.gpt import GPT, GPTConfig


def _random_qkv(B=2, H=4, T=32, D=16):
    g = torch.Generator().manual_seed(7)
    return (torch.randn(B, H, T, D, generator=g),
            torch.randn(B, H, T, D, generator=g),
            torch.randn(B, H, T, D, generator=g))


def test_shapes():
    q, k, v = _random_qkv()
    y = attention_vanilla(q, k, v, causal_mask(q.size(2), "cpu"))
    assert y.shape == q.shape


def test_no_future_leakage():
    """Rows must not depend on FUTURE tokens: perturb tokens j>i and
    check rows <= i are bit-identical (softmax(-inf) contributes exactly 0)."""
    q, k, v = _random_qkv(T=16)
    mask = causal_mask(16, "cpu")
    y1 = attention_vanilla(q, k, v, mask)
    v2 = v.clone()
    v2[:, :, 8:, :] += 100.0       # wreak havoc on positions 8..15
    k2 = k.clone()
    k2[:, :, 8:, :] += 100.0
    y2 = attention_vanilla(q, k2, v2, mask)
    # Rows 0..7 may only see positions <= row index, so they are unchanged.
    assert torch.allclose(y1[:, :, :8, :], y2[:, :, :8, :], atol=1e-6)


def test_tiled_matches_vanilla():
    """FlashAttention algorithm 1 (tiled) must equal the naked math."""
    q, k, v = _random_qkv(T=64)
    mask = causal_mask(64, "cpu")
    y_vanilla = attention_vanilla(q, k, v, mask)
    y_tiled = flash_attention_tiled(q, k, v, mask, tile_size=16)
    assert torch.allclose(y_vanilla, y_tiled, atol=1e-4, rtol=1e-4)


def test_fused_matches_vanilla():
    """PyTorch's fused SDPA path must equal the naked math."""
    q, k, v = _random_qkv(T=64)
    mask = causal_mask(64, "cpu")
    y_vanilla = attention_vanilla(q, k, v, mask)
    y_fused = flash_attention_fused(q, k, v)
    assert torch.allclose(y_vanilla, y_fused, atol=1e-3, rtol=1e-3)


def test_module_backends_identical_with_rope():
    """Same weights, rope position, vanilla vs fused: outputs must match -
    proving the backend swap changes nothing but the computation path."""
    torch.manual_seed(0)
    cfg_vanilla = GPTConfig(vocab_size=300, block_size=32, n_layer=2,
                            n_head=2, n_embd=32, attention="vanilla",
                            position="rope")
    cfg_fused = GPTConfig(vocab_size=300, block_size=32, n_layer=2,
                          n_head=2, n_embd=32, attention="flash_fused",
                          position="rope")
    m1 = GPT(cfg_vanilla)
    m2 = GPT(cfg_fused)
    m2.load_state_dict(m1.state_dict())
    x = torch.randint(0, 300, (2, 32))
    with torch.no_grad():
        l1, _ = m1(x)
        l2, _ = m2(x)
    assert torch.allclose(l1, l2, atol=1e-5)


def test_module_backends_identical_with_learned():
    """Same check with the learned position embedding (experiments A/C)."""
    torch.manual_seed(0)
    cfg_a = GPTConfig(vocab_size=300, block_size=32, n_layer=2, n_head=2,
                      n_embd=32, attention="vanilla", position="learned")
    cfg_c = GPTConfig(vocab_size=300, block_size=32, n_layer=2, n_head=2,
                      n_embd=32, attention="flash_fused", position="learned")
    m1, m2 = GPT(cfg_a), GPT(cfg_c)
    m2.load_state_dict(m1.state_dict())
    x = torch.randint(0, 300, (2, 32))
    with torch.no_grad():
        l1, _ = m1(x)
        l2, _ = m2(x)
    assert torch.allclose(l1, l2, atol=1e-5)


def test_tiled_backend_module_matches_vanilla_module():
    """The pure-PyTorch tiled backend in the full module must also match."""
    torch.manual_seed(0)
    cfg_a = GPTConfig(vocab_size=300, block_size=32, n_layer=2, n_head=2,
                      n_embd=32, attention="vanilla", position="learned")
    cfg_t = GPTConfig(vocab_size=300, block_size=32, n_layer=2, n_head=2,
                      n_embd=32, attention="flash_tiled", position="learned",
                      flash_tile_size=8)
    m1, m2 = GPT(cfg_a), GPT(cfg_t)
    m2.load_state_dict(m1.state_dict())
    x = torch.randint(0, 300, (2, 32))
    with torch.no_grad():
        l1, _ = m1(x)
        l2, _ = m2(x)
    assert torch.allclose(l1, l2, atol=1e-4)


def test_tiled_no_future_leakage():
    """Causality holds in the tiled backend too."""
    q, k, v = _random_qkv(T=32)
    mask = causal_mask(32, "cpu")
    y1 = flash_attention_tiled(q, k, v, mask, tile_size=8)
    v2 = v.clone()
    v2[:, :, 20:, :] += 100.0
    y2 = flash_attention_tiled(q, k, v2, mask, tile_size=8)
    assert torch.allclose(y1[:, :, :20, :], y2[:, :, :20, :], atol=1e-5)
