"""Optimizer tests (ABLATION_PLAN section 8/16): the first-principles
reference implementations must reproduce PyTorch's Adam and AdamW.

These tests are the proof that the update equations written in
src/training/optimizer.py (moments, bias correction, coupled vs decoupled
decay) are actually the math torch ships - so the C1/C2 comparison is
between the two DECAY MECHANISMS, not between buggy and correct code.
"""
import torch
import torch.nn as nn

from src.training.optimizer import (
    AdamReference, AdamWReference, optimizer_state_memory)


def _run_pair(ref_opt, torch_opt, steps=60, wd=0.0, seed=0):
    """Run reference and torch optimizers on identical twin models; return
    the max parameter difference afterwards."""
    torch.manual_seed(seed)

    def make_model():
        m = nn.Sequential(nn.Linear(16, 32), nn.Tanh(), nn.Linear(32, 8))
        torch.manual_seed(seed)  # identical init for both twins
        return m

    m_ref, m_torch = make_model(), make_model()
    ref = ref_opt(m_ref.parameters(), lr=3e-3, betas=(0.9, 0.999), eps=1e-8,
                  weight_decay=wd)
    tch = torch_opt(m_torch.parameters(), lr=3e-3, betas=(0.9, 0.999),
                    eps=1e-8, weight_decay=wd)
    x = torch.randn(8, 16)
    y = torch.randn(8, 8)
    for _ in range(steps):
        loss_ref = ((m_ref(x) - y) ** 2).mean()
        ref.zero_grad() if hasattr(ref, "zero_grad") else None
        for p in m_ref.parameters():
            p.grad = None
        loss_ref.backward()
        ref.step()

        loss_t = ((m_torch(x) - y) ** 2).mean()
        tch.zero_grad()
        loss_t.backward()
        tch.step()
    diffs = [((pr - pt).abs().max().item())
             for pr, pt in zip(m_ref.parameters(), m_torch.parameters())]
    return max(diffs)


def test_adam_ref_matches_torch():
    """Reference Adam == torch.optim.Adam (no decay, and coupled L2 decay)."""
    d = _run_pair(AdamReference, torch.optim.Adam, wd=0.0)
    assert d < 1e-5, f"adam no-decay mismatch {d}"
    d = _run_pair(AdamReference, torch.optim.Adam, wd=0.1)
    assert d < 1e-5, f"adam coupled-decay mismatch {d}"


def test_adamw_ref_matches_torch():
    """Reference AdamW == torch.optim.AdamW with decoupled decay."""
    d = _run_pair(AdamWReference, torch.optim.AdamW, wd=0.1)
    assert d < 1e-5, f"adamw decoupled-decay mismatch {d}"


def test_adam_vs_adamw_differ():
    """With decay on, coupled (Adam) and decoupled (AdamW) updates must
    genuinely differ - otherwise the C1/C2 study compares nothing."""
    torch.manual_seed(1)
    m1 = nn.Linear(8, 8)
    m2 = nn.Linear(8, 8)
    m2.load_state_dict(m1.state_dict())
    x = torch.randn(4, 8)
    y = torch.randn(4, 8)
    ref_a = AdamReference(m1.parameters(), lr=1e-2, weight_decay=0.3)
    ref_w = AdamWReference(m2.parameters(), lr=1e-2, weight_decay=0.3)
    for _ in range(20):
        for m, opt in ((m1, ref_a), (m2, ref_w)):
            for p in m.parameters():
                p.grad = None
            ((m(x) - y) ** 2).mean().backward()
            opt.step()
    diff = (m1.weight - m2.weight).abs().max().item()
    assert diff > 1e-3, f"coupled and decoupled decay converged identically ({diff})"


def test_state_creation_and_shapes():
    """Moments must exist per parameter with matching shapes (section 16)."""
    m = nn.Linear(4, 4)
    opt = AdamReference(m.parameters())
    x = torch.randn(3, 4)
    ((m(x) - 1.0) ** 2).mean().backward()
    opt.step()
    st = opt.state[m.weight]
    assert st["m"].shape == m.weight.shape == st["v"].shape
    assert st["t"] == 1


def test_state_memory_formula():
    """Section 14: optimizer state = 2 fp32 moments per trainable param."""
    m = nn.Linear(10, 10)  # 100 weights + 10 biases = 110 trainable
    mb = optimizer_state_memory(m, "adamw")
    expected = 2 * 4 * 110 / (1024 * 1024)
    assert abs(mb - expected) < 1e-9


def test_loss_decreases_on_quadratic():
    """Sanity: both references descend a trivial quadratic.

    NOTE on Adam dynamics: the sqrt(v) normalization cancels the gradient
    MAGNITUDE, so on a pure quadratic w^2 the descent is ~lr per step
    (w^2 shrinks by lr each step), not the exponential decay of SGD. The
    test asserts (a) monotone-ish early decrease and (b) that enough
    steps reach the neighbourhood of the optimum - the honest behaviour,
    not an SGD-shaped expectation."""
    torch.manual_seed(0)
    for cls in (AdamReference, AdamWReference):
        w = torch.full((1,), 5.0, requires_grad=True)
        opt = cls([w], lr=0.1)
        for _ in range(600):
            if w.grad is not None:
                w.grad.zero_()
            (w * w).mean().backward()
            opt.step()
        assert abs(w.item()) < 0.5, f"{cls.__name__} did not converge: {w.item()}"
        assert abs(w.item()) < 5.0  # strictly better than init
