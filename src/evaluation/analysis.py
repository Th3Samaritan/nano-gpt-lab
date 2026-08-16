"""Result aggregation: comparison tables + the layman report suite.

WHAT
----
collect_summaries() scans runs/{dataset}/*/ for config.yaml + metrics.csv
and rebuilds one summary dict per finished run (best val loss, perplexity,
speed, peak memory, params, run dir). comparison_table() renders that as a
text table; render_layman_reports() renders the PNG suite from
src/utils/plots.py into a results directory.

WHY
----
Runs live in files, not variables; anything that survives a session must
be reconstructable from the run directory alone. The analysis layer is
pure read-only aggregation over metrics.csv + config.yaml - it never
re-trains and never guesses.

WHERE
-----
Used by scripts/analyze.py and notebooks/04_experiment_analysis.ipynb,
both on the local machine and after a Colab/Kaggle session (the runs directory is
mirrored to Drive / persisted on Kaggle).
"""
from __future__ import annotations

import csv
import json
import os
from typing import List, Optional

from src.utils.environment import runs_dir
from src.utils.plots import plot_all_comparisons


def _read_metrics(metrics_csv: str) -> dict:
    """Aggregate a metrics.csv into best/throughput/memory scalars."""
    out = {"best_val_loss": None, "final_tokens_per_sec": None,
           "peak_gpu_mem_mb": None, "last_tokens_seen": None,
           "rows": 0}
    if not os.path.exists(metrics_csv):
        return out
    with open(metrics_csv, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            try:
                val = float(row["val_loss"])
            except (ValueError, KeyError):
                continue
            out["rows"] += 1
            if out["best_val_loss"] is None or val < out["best_val_loss"]:
                out["best_val_loss"] = val
            # Each field parses independently: rows may legitimately carry
            # blank cells (e.g. the final row has no train_loss), and one
            # blank field must never discard the others.
            for key in ("tokens_seen",):
                try:
                    out["last_tokens_seen"] = int(row[key])
                except (ValueError, KeyError):
                    pass
            try:
                out["final_tokens_per_sec"] = float(row["tokens_per_sec"])
            except (ValueError, KeyError):
                pass
            try:
                out["peak_gpu_mem_mb"] = float(row["gpu_mem_mb"])
            except (ValueError, KeyError):
                pass
    return out


def collect_summaries(dataset: str, size: Optional[str] = None,
                      seeds: Optional[List[int]] = None) -> List[dict]:
    """One summary dict per completed run matching the filters."""
    from src.utils.config import load_yaml
    base = os.path.join(runs_dir(), dataset)
    summaries: List[dict] = []
    if not os.path.isdir(base):
        return summaries
    for name in sorted(os.listdir(base)):
        run_dir = os.path.join(base, name)
        cfg_path = os.path.join(run_dir, "config.yaml")
        if not os.path.exists(cfg_path):
            continue
        cfg = load_yaml(cfg_path)
        if size and cfg.get("size") != size:
            continue
        if seeds and cfg.get("seed") not in seeds:
            continue
        agg = _read_metrics(os.path.join(run_dir, "metrics.csv"))
        if agg["best_val_loss"] is None:
            continue  # never evaluated: not a completed run
        import math
        summaries.append({
            "run_dir": run_dir,
            "dataset": cfg.get("dataset", dataset),
            "experiment": cfg.get("experiment", name),
            "attention": cfg.get("attention"),
            "position": cfg.get("position"),
            "size": cfg.get("size"),
            "seed": cfg.get("seed"),
            "best_val_loss": round(agg["best_val_loss"], 4),
            "val_ppl": round(math.exp(agg["best_val_loss"]), 2),
            "tokens_per_sec": round(agg["final_tokens_per_sec"] or 0.0, 1),
            "peak_gpu_mem_mb": round(agg["peak_gpu_mem_mb"] or 0.0, 1),
            "tokens_seen": agg["last_tokens_seen"] or 0,
        })
    return summaries


def comparison_table(summaries: List[dict]) -> str:
    """Plain-text table (plan section 17's required columns)."""
    if not summaries:
        return "(no completed runs found)"
    header = (f"{'variant':<22} {'seed':>5} {'val loss':>9} {'ppl':>8} "
              f"{'tok/s':>9} {'peakMB':>8} {'tokens':>12}")
    lines = [header, "-" * len(header)]
    for s in sorted(summaries, key=lambda d: (d["seed"], str(d["experiment"]))):
        lines.append(
            f"{s['attention']}+{s['position']:<13} {s['seed']:>5} "
            f"{s['best_val_loss']:>9.4f} {s['val_ppl']:>8.2f} "
            f"{s['tokens_per_sec']:>9.1f} {s['peak_gpu_mem_mb']:>8.1f} "
            f"{s['tokens_seen']:>12,}")
    return "\n".join(lines)


def save_comparison_json(summaries: List[dict], path: str) -> None:
    """Machine-readable aggregation (kept alongside every report)."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(summaries, f, indent=2)


def render_layman_reports(summaries: List[dict], dataset: str,
                          out_dir: str, threshold: float = 2.0) -> dict:
    """Render the full PNG suite; returns {chart_name: path}."""
    return plot_all_comparisons(
        summaries, out_dir, dataset_name=dataset, threshold=threshold)


def analyze(dataset: str, size: Optional[str] = None,
            seeds: Optional[List[int]] = None, threshold: float = 2.0,
            out_dir: Optional[str] = None) -> dict:
    """Convenience wrapper: collect + table + json + PNG suite."""
    summaries = collect_summaries(dataset, size, seeds)
    if out_dir is None:
        from src.utils.environment import repo_root
        out_dir = os.path.join(repo_root(), "results", dataset)
    save_comparison_json(summaries, os.path.join(out_dir, "comparison.json"))
    plots = render_layman_reports(summaries, dataset, out_dir, threshold)
    return {"summaries": summaries, "plots": plots,
            "table": comparison_table(summaries)}
