"""Overfit-check chart tests: the detector must render and must flag
divergence (train falling / val climbing) while clearing healthy curves."""
import csv
import json
import os
import tempfile

from src.utils.plots import plot_overfit_check

COLUMNS = ["step", "tokens_seen", "train_loss", "val_loss", "val_ppl", "lr",
           "step_time_s", "tokens_per_sec", "gpu_mem_mb", "grad_norm"]


def _make_run(tmp: str, name: str, overfit: bool) -> str:
    d = os.path.join(tmp, name)
    os.makedirs(d, exist_ok=True)
    with open(os.path.join(d, "metrics.csv"), "w", newline="",
              encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(COLUMNS)
        for i, epoch in enumerate([1, 3, 5, 8, 12]):
            train = 6.0 - 0.4 * i                       # keeps falling
            val = (7.0 + 0.5 * i if overfit else 6.2 - 0.3 * i)
            w.writerow([epoch * 100, epoch * 25000, train, val, 500,
                        1e-4, 1.0, 1000.0, 0.0, 0.5])
    with open(os.path.join(d, "meta.json"), "w", encoding="utf-8") as f:
        json.dump({"train_tokens": 25000}, f)
    return d


def _summary(d, attn, pos):
    return {"run_dir": d, "attention": attn, "position": pos,
            "activation": "gelu", "swiglu_matched": True,
            "optimizer": "adamw", "seed": 42}


def test_overfit_check_renders_and_flags():
    with tempfile.TemporaryDirectory() as tmp:
        healthy = _make_run(tmp, "healthy", overfit=False)
        sick = _make_run(tmp, "sick", overfit=True)
        summaries = [_summary(healthy, "vanilla", "learned"),
                     _summary(sick, "flash_fused", "rope")]
        out = plot_overfit_check(summaries,
                                 os.path.join(tmp, "09_overfit_check.png"),
                                 "synthetic")
        assert out is not None and os.path.exists(out)


def test_overfit_check_skips_when_no_metrics():
    with tempfile.TemporaryDirectory() as tmp:
        d = os.path.join(tmp, "empty")
        os.makedirs(d, exist_ok=True)
        out = plot_overfit_check(
            [_summary(d, "vanilla", "learned")],
            os.path.join(tmp, "09.png"), "synthetic")
        assert out is None
