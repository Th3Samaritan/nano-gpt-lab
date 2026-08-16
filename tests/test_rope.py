"""RoPE tests (plan section 10): the mathematical properties that make
rotation-based positions correct, not just shapes that happen to run.

The defining property of RoPE is relative-position invariance:
    <R(m) q, R(n) k>  =  <R(m-n) q, k>
If this fails, positions leak into attention in a wrong way. The other
test checks orthogonality: rotation must not change vector lengths, or
attention scores would be distorted by position alone.
"""
import torch

from src.model.rope import apply_rope, build_rope_tables, rotate_half


def test_table_shapes():
    cos, sin = build_rope_tables(head_dim=32, max_seq_len=100)
    assert cos.shape == (100, 32)
    assert sin.shape == (100, 32)


def test_apply_shapes_multiple_lengths():
    """RoPE must work for any sequence length up to the table size."""
    cos, sin = build_rope_tables(32, 64)
    for T in (1, 5, 17, 64):
        q = torch.randn(2, 4, T, 32)
        k = torch.randn(2, 4, T, 32)
        q2, k2 = apply_rope(q, k, cos[:T], sin[:T])
        assert q2.shape == q.shape and k2.shape == k.shape


def test_rotation_preserves_norm():
    """Orthogonality: rotating a vector keeps its length exactly."""
    cos, sin = build_rope_tables(16, 10)
    x = torch.randn(3, 5, 16)
    rotated = x * cos[:5] + rotate_half(x) * sin[:5]
    assert torch.allclose(rotated.norm(dim=-1), x.norm(dim=-1), atol=1e-5)


def test_relative_position_property():
    """<R(m) q, R(n) k> must equal <R(m-n) q, k> - the core of RoPE."""
    torch.manual_seed(3)
    hd = 16
    cos, sin = build_rope_tables(hd, 40)
    q0 = torch.randn(1, 1, 1, hd)
    k0 = torch.randn(1, 1, 1, hd)

    def dot_rotated(x, y, i, j):
        xi = x * cos[i:i + 1] + rotate_half(x) * sin[i:i + 1]
        yj = y * cos[j:j + 1] + rotate_half(y) * sin[j:j + 1]
        return (xi * yj).sum()

    for m, n in [(0, 0), (3, 3), (2, 9), (15, 7), (30, 1)]:
        lhs = dot_rotated(q0, k0, m, n)
        rhs = dot_rotated(q0, k0, m - n, 0) if m >= n else \
            dot_rotated(q0, k0, 0, n - m)
        # relative rotation of q and k with opposite sign is equivalent
        assert torch.allclose(lhs, rhs, atol=1e-4), f"failed at m={m}, n={n}"


def test_zero_position_is_identity():
    """Position 0 applies no rotation."""
    cos, sin = build_rope_tables(16, 8)
    x = torch.randn(4, 16)
    rotated = x * cos[0] + rotate_half(x) * sin[0]
    assert torch.allclose(rotated, x, atol=1e-6)


def test_deterministic():
    """Same input, same table -> bit-identical output."""
    torch.manual_seed(0)
    cos, sin = build_rope_tables(8, 16)
    q = torch.randn(1, 2, 16, 8)
    k = torch.randn(1, 2, 16, 8)
    q1, k1 = apply_rope(q, k, cos, sin)
    q2, k2 = apply_rope(q, k, cos, sin)
    assert torch.equal(q1, q2) and torch.equal(k1, k2)
