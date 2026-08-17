"""FFN/activation tests (ABLATION_PLAN section 16): shapes, gradients,
parameter accounting, and the SwiGLU fairness rule.

The fairness test is the scientific core of Study B: the param-matched
SwiGLU must carry (approximately) the same FFN parameters as GELU, so any
quality difference between B1 and B4a is attributable to the GATE, not to
a silently larger model.
"""
import torch

from src.model.gpt import GPT, GPTConfig


def _cfg(activation, swiglu_matched=True, **kw):
    base = dict(vocab_size=300, block_size=32, n_layer=2, n_head=2,
                n_embd=32, attention="vanilla", position="learned",
                activation=activation, swiglu_matched=swiglu_matched)
    base.update(kw)
    return GPTConfig(**base)


def test_all_activations_forward_backward():
    """Every Study B variant must run forward + backward with the same
    input/output shapes (the training loop must never know which FFN)."""
    torch.manual_seed(0)
    for act, matched in [("gelu", True), ("relu", True), ("silu", True),
                         ("swiglu", True), ("swiglu", False)]:
        model = GPT(_cfg(act, matched))
        x = torch.randint(0, 300, (2, 32))
        y = torch.randint(0, 300, (2, 32))
        logits, loss = model(x, y)
        assert logits.shape == (2, 32, 300), act
        loss.backward()
        missing = [n for n, p in model.named_parameters()
                   if p.grad is None and p.requires_grad]
        assert missing == [], f"no gradient for {missing} ({act})"


def test_swiglu_matched_param_count():
    """Fairness rule (ABLATION_PLAN section 5): matched SwiGLU must carry
    (approximately) the GELU baseline's FFN parameters.

    Weight accounting: GELU FFN weights = n*4n + 4n*n = 8n^2.
    Matched SwiGLU (h = round(8n/3)): 3*n*h ~= 8n^2, so the count differs
    only by the rounding of h and the (small) bias term."""
    n = 32
    gelu = GPT(_cfg("gelu")).param_breakdown()
    swiglu = GPT(_cfg("swiglu", True)).param_breakdown()
    diff = abs(swiglu["ffn"] - gelu["ffn"])
    # h = round(256/3) = 85 vs ideal 85.33 -> weight delta ~ n*0.33*3 + bias
    # delta - allow a generous 2% of the FFN budget (still ~1/10 of the
    # native SwiGLU's extra cost).
    assert diff <= 0.02 * gelu["ffn"], \
        f"matched SwiGLU FFN differs by {diff} params ({gelu['ffn']} vs {swiglu['ffn']})"
    # and the difference is tiny in absolute terms for this model size
    assert diff < 500, f"matched SwiGLU off by {diff} params"


def test_swiglu_native_costs_more():
    """Architecture-native SwiGLU (h=4n) carries ~1.5x the GELU FFN weights:
    the whole point of reporting both interpretations separately."""
    gelu = GPT(_cfg("gelu")).param_breakdown()["ffn"]
    native = GPT(_cfg("swiglu", False)).param_breakdown()["ffn"]
    assert native > 1.3 * gelu, f"native SwiGLU should cost more: {native} vs {gelu}"


def test_scalar_activations_share_width():
    """GELU/ReLU/SiLU are scalar swaps: identical FFN parameter counts."""
    counts = [GPT(_cfg(a)).param_breakdown()["ffn"] for a in ("gelu", "relu", "silu")]
    assert counts[0] == counts[1] == counts[2], counts


def test_param_breakdown_sums_to_total():
    """Section 15 accounting must be internally consistent."""
    for act in ("gelu", "relu", "silu", "swiglu"):
        b = GPT(_cfg(act)).param_breakdown()
        parts = b["embedding"] + b["attention"] + b["ffn"] + b["lm_head"] + b["other"]
        assert parts == b["total"], act
        assert b["lm_head"] == 0  # weight-tied: no extra head parameters
