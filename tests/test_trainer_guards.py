"""Training-guard tests: the overfit defenses that the Kaggle gpt2 run
exposed as necessary (65M tokens over a 250k-token corpus = 260 epochs =
memorization + validation divergence).

These helpers are pure functions so their rules are pinned down exactly:
the epoch cap, the early-stop decision, and the overfit signature."""
from src.training.trainer import (
    early_stop_reached, epochs_from_tokens, overfit_signal)


def test_epochs_from_tokens():
    assert abs(epochs_from_tokens(65_536_000, 250_000) - 262.14) < 0.1
    assert epochs_from_tokens(0, 250_000) == 0.0
    assert epochs_from_tokens(1000, 0) == 1000.0  # unknown size: no crash


def test_early_stop_reached():
    assert early_stop_reached(10, 10) is True
    assert early_stop_reached(9, 10) is False
    assert early_stop_reached(100, 0) is False  # patience 0 disables it


def test_overfit_signal():
    assert overfit_signal(12.4, 8.2) is True    # the Kaggle collapse case
    assert overfit_signal(6.2, 6.3) is False
    assert overfit_signal(8.4, 8.2, threshold=0.1) is True  # custom bar
