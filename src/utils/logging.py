"""Run logger: train.log + metrics.csv + loss-curve plotting.

WHAT
----
RunLogger writes two artifacts inside every run directory:
    train.log   - human readable, timestamped console history
    metrics.csv - machine readable, one row per evaluation point:
                  step, tokens_seen, train_loss, val_loss, val_ppl, lr,
                  step_time_s, tokens_per_sec, gpu_mem_mb, grad_norm
plot_loss_curves() renders train/val loss from metrics.csv to PNG.

WHY
----
The plan requires plain CSV/JSON logging that survives without any external
tracker (section 22), and warns against judging experiments by final loss
alone (section 18) - learning curves over TOKENS (not steps) are the primary
evidence, because variants that spend different wall-clock time per step must
be compared on an equal-token axis.

WHERE
-----
Used by training/trainer.py and by scripts + the analysis notebook.
"""
from __future__ import annotations

import csv
import os
import time

METRIC_COLUMNS = [
    "step", "tokens_seen", "train_loss", "val_loss", "val_ppl", "lr",
    "step_time_s", "tokens_per_sec", "gpu_mem_mb", "grad_norm",
]


class RunLogger:
    """Writes train.log and metrics.csv inside one run directory."""

    def __init__(self, run_dir: str, resume: bool = False):
        self.run_dir = run_dir
        os.makedirs(run_dir, exist_ok=True)
        self.log_path = os.path.join(run_dir, "train.log")
        self.csv_path = os.path.join(run_dir, "metrics.csv")
        if not resume:
            # Fresh run: truncate both files so stale data never leaks in.
            open(self.log_path, "w", encoding="utf-8").close()
            with open(self.csv_path, "w", newline="", encoding="utf-8") as f:
                csv.writer(f).writerow(METRIC_COLUMNS)

    def log(self, message: str) -> None:
        """Append a timestamped line to train.log and echo it to console."""
        line = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {message}"
        with open(self.log_path, "a", encoding="utf-8") as f:
            f.write(line + "\n")
        print(line)

    def record_metrics(self, row: dict) -> None:
        """Append one evaluation row to metrics.csv (column order enforced)."""
        with open(self.csv_path, "a", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow([row.get(col, "") for col in METRIC_COLUMNS])


def plot_loss_curves(run_dir: str, out_path: str | None = None) -> str | None:
    """Plot train/val loss vs tokens from a metrics.csv into a PNG.

    X axis is tokens_seen (not steps) so curves from different attention
    implementations - which run at different speeds - remain comparable.
    Returns the output PNG path, or None when there is nothing to plot.
    """
    import matplotlib
    matplotlib.use("Agg")  # headless-safe: never opens a window
    import matplotlib.pyplot as plt

    csv_path = os.path.join(run_dir, "metrics.csv")
    if not os.path.exists(csv_path):
        return None
    steps, tokens, train_l, val_l = [], [], [], []
    with open(csv_path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            try:
                steps.append(int(row["step"]))
                tokens.append(int(row["tokens_seen"]))
                train_l.append(float(row["train_loss"]))
                val_l.append(float(row["val_loss"]))
            except (ValueError, KeyError):
                continue
    if not steps:
        return None
    if out_path is None:
        out_path = os.path.join(run_dir, "loss_curves.png")
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(tokens, train_l, label="train loss")
    ax.plot(tokens, val_l, label="val loss")
    ax.set_xlabel("tokens seen")
    ax.set_ylabel("loss")
    ax.set_title("loss curves (x = tokens, so variants are comparable)")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    return out_path
