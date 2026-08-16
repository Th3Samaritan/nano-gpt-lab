"""Transformer assembly tests: shapes for every variant, weight tying,
parameter accounting, generation behaviour, and gradient flow.

The shape matrix below runs ALL FOUR experiments (A/B/C/D) through the
same assertions - if any variant cannot do these basics, it is excluded
from the ablation before a single training step.
"""
import torch

from src.model.gpt import GPT, GPTConfig

VARIANTS = [
    ("vanilla", "learned"),   # A
    ("vanilla", "rope"),      # B
    ("flash_fused", "learned"),  # C
    ("flash_fused", "rope"),  # D
    ("flash_tiled", "rope"),  # first-principles tiled path too
]


def _cfg(attention, position, **kw):
    base = dict(vocab_size=300, block_size=32, n_layer=2, n_head=2,
                n_embd=32, attention=attention, position=position)
    base.update(kw)
    return GPTConfig(**base)


def test_output_shapes_all_variants():
    torch.manual_seed(0)
    for attn, pos in VARIANTS:
        model = GPT(_cfg(attn, pos))
        x = torch.randint(0, 300, (2, 32))
        y = torch.randint(0, 300, (2, 32))
        logits, loss = model(x, y)
        assert logits.shape == (2, 32, 300), f"{attn}+{pos}"
        assert loss is not None and loss.ndim == 0, f"{attn}+{pos}"


def test_weight_tying():
    """lm_head must BE the wte matrix (GPT-2/GPT-3 weight tying)."""
    model = GPT(_cfg("vanilla", "learned"))
    assert model.lm_head_weight is model.wte.weight


def test_exact_param_count():
    """Hand-computed parameter count for the 2L/2H/32 test config."""
    model = GPT(_cfg("vanilla", "learned"))
    # wte 300*32 + wpe 32*32
    # per block: c_attn 32*96+96, c_proj 32*32+32,
    #            c_fc 32*128+128, c_proj 128*32+32, ln1 64, ln2 64
    # final ln_f 64
    expected = 300 * 32 + 32 * 32 + 2 * (3168 + 1056 + 4224 + 4128 + 128) + 64
    assert model.get_num_params(non_embedding=False) == expected
    assert model.get_num_params(non_embedding=True) == \
        expected - 300 * 32 - 32 * 32


def test_rope_has_no_position_table():
    """RoPE must NOT add the learned wpe parameters (clean ablation)."""
    m_learned = GPT(_cfg("vanilla", "learned"))
    m_rope = GPT(_cfg("vanilla", "rope"))
    assert m_rope.wpe is None
    diff = (m_learned.get_num_params(non_embedding=False)
            - m_rope.get_num_params(non_embedding=False))
    assert diff == 32 * 32  # exactly one block_size x n_embd table


def test_generate_respects_block_size():
    """Generation longer than context must still work (sliding window)."""
    model = GPT(_cfg("vanilla", "rope"))
    x = torch.randint(0, 300, (1, 32))
    out = model.generate(x, max_new_tokens=40, temperature=0.8, top_k=10)
    assert out.shape == (1, 72)
    assert out.max().item() < 300


def test_gradients_flow():
    """Backward must reach every trainable parameter (no dead wiring)."""
    model = GPT(_cfg("flash_fused", "rope"))
    x = torch.randint(0,300,(2,32)); y = torch.randint(0,300,(2,32))
    _, loss = model(x, y)
    loss.backward()
    missing = [n for n, p in model.named_parameters()
               if p.grad is None and p.requires_grad]
    assert missing == [], f"no gradient for {missing}"


def test_generate_all_variants_run():
    """Sampling smoke test across the whole matrix."""
    torch.manual_seed(0)
    for attn, pos in VARIANTS:
        model = GPT(_cfg(attn, pos))
        x = torch.randint(0, 300, (1, 8))
        out = model.generate(x, max_new_tokens=5, temperature=1.0)
        assert out.shape == (1, 13), f"{attn}+{pos}"


def test_parallel_wiring_on_cpu():
    """Multi-GPU wiring must be a no-op without CUDA: the wrapper returns
    the plain model and get_base_gpt recovers it (the T4x2 path uses the
    same code, so this guards the interface the trainer relies on)."""
    from src.utils.device import get_base_gpt, resolve_device_ids, wrap_parallel
    model = GPT(_cfg("flash_fused", "rope"))
    ids = resolve_device_ids("auto")  # [] on CPU-only machines
    wrapped, base = wrap_parallel(model, ids)
    assert wrapped is model and base is model
    assert get_base_gpt(wrapped) is model
    # And the DP-safe loss path the trainer now uses everywhere:
    import torch.nn.functional as F
    x = torch.randint(0, 300, (2, 32))
    y = torch.randint(0, 300, (2, 32))
    out = wrapped(x)
    logits = out[0] if isinstance(out, tuple) else out
    loss = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1))
    assert loss.ndim == 0 and loss.item() > 0
