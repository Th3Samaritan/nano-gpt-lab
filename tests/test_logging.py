"""Plotting regression tests: the metrics.csv edge cases that must NEVER
crash a completed run's reporting.

The specific case here bit us on Kaggle: the trainer appends a final
evaluation row that carries val_loss but no train_loss, and the old curve
plotters built one shared token list - crashing with mismatched lengths
AFTER training finished. These tests pin that bug down for both plotters.
"""
import csv
import os
import tempfile

from src.utils.logging import plot_loss_curves
from src.utils.plots import plot_loss_curves_layman

COLUMNS = ["step", "tokens_seen", "train_loss", "val_loss", "val_ppl", "lr",
           "step_time_s", "tokens_per_sec", "gpu_mem_mb", "grad_norm"]


def _write_metrics(dir_path, include_final_row_without_train=True):
    path = os.path.join(dir_path, "metrics.csv")
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(COLUMNS)
        for step, tok, t, v in [
                (1, 8192, 8.29, 8.30),
                (100, 819200, 7.22, 7.25),
                (200, 1638400, 6.34, 6.38)]:
            w.writerow([step, tok, t, v, 500, 3e-4, 2.7, 2800.0, 0.0, 0.9])
        if include_final_row_without_train:
            # the trainer's final eval row: val_loss present, train_loss blank
            w.writerow([250, 2048000, "", 6.18, 485, 9e-5, "", 3000.0, 0.0, ""])
    return path


def test_plot_loss_curves_final_row_without_train_loss():
    """The Kaggle crash case: plotting must succeed, not raise."""
    with tempfile.TemporaryDirectory() as tmp:
        _write_metrics(tmp, include_final_row_without_train=True)
        out = plot_loss_curves(tmp)
        assert out is not None and os.path.exists(out)


def test_plot_loss_curves_layman_final_row_without_train_loss():
    with tempfile.TemporaryDirectory() as tmp:
        _write_metrics(tmp, include_final_row_without_train=True)
        out = plot_loss_curves_layman(tmp)
        assert out is not None and os.path.exists(out)


def test_plot_loss_curves_all_rows_complete():
    with tempfile.TemporaryDirectory() as tmp:
        _write_metrics(tmp, include_final_row_without_train=False)
        out = plot_loss_curves(tmp)
        assert out is not None and os.path.exists(out)


def test_plot_loss_curves_empty_returns_none():
    with tempfile.TemporaryDirectory() as tmp:
        assert plot_loss_curves(tmp) is None  # no metrics.csv at all
