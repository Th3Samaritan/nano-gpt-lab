"""Metric tests: the industry battery must be mathematically honest.

Each test checks the FORMULA on synthetic data where the answer is known,
so a chart can never silently show a miscalculated number.
"""
import math

import torch

from src.evaluation.metrics import (
    bits_per_byte, distinct_n, expected_calibration_error, repetition_rate)


def test_bits_per_byte_formula():
    """bpb = loss / ln(2) * tokens_per_char."""
    loss, tpc = 3.0, 0.5
    assert abs(bits_per_byte(loss, tpc) - 3.0 / math.log(2) * 0.5) < 1e-9


def test_ece_perfect_calibration_is_zero():
    """If every bin's accuracy equals its confidence, ECE must be 0."""
    # 60% of samples at conf 0.6 with 60% correct; 40% at 0.9 with 90% correct
    confs = torch.cat([torch.full((600,), 0.6), torch.full((400,), 0.9)])
    corrects = torch.cat([torch.ones(600,), torch.ones(400,)])
    corrects[:240] = 0.0    # first 600: 360/600 = 0.6 correct
    corrects[-40:] = 0.0    # last 400: 360/400 = 0.9 correct
    ece = expected_calibration_error(confs, corrects, n_bins=15)
    assert ece < 0.02, f"ECE should be ~0, got {ece}"


def test_ece_overconfidence_is_positive():
    """Always 90% confident but only 50% right => ECE clearly > 0."""
    confs = torch.full((1000,), 0.9)
    corrects = torch.cat([torch.ones(500), torch.zeros(500)])
    ece = expected_calibration_error(confs, corrects, n_bins=15)
    assert ece > 0.2, f"overconfident model should have high ECE, got {ece}"


def test_distinct_n():
    assert distinct_n([1, 2, 3, 4], 1) == 1.0      # all unique
    assert distinct_n([1, 1, 1, 1], 1) == 0.25     # all repeats
    # bigrams of [1,2,1,2]: (1,2),(2,1),(1,2) -> 2 unique of 3 total
    assert abs(distinct_n([1, 2, 1, 2], 2) - 2 / 3) < 1e-9


def test_repetition_rate():
    assert repetition_rate([1, 1, 2, 2, 2, 3]) == 3 / 6
    assert repetition_rate([1, 2, 3]) == 0.0
    assert repetition_rate([]) == 0.0
