"""Tiny Shakespeare preparation (Dataset 1 - the pipeline debugger).

WHAT
----
Downloads Karpathy's tinyshakespeare corpus (~1.1MB, 40k lines of
Shakespeare plays), splits it 90/10 by characters, then runs the shared
BPE + tokenize pipeline (data/download.py).

WHY
----
Per the plan (section 4) this dataset exists to verify the whole pipeline,
debug the transformer, and run rapid A/B/C/D architecture experiments -
not for benchmark quality. It is small enough to prepare in seconds on a
CPU, which makes it the default dataset for the LOCAL ZBook tier.

WHERE
-----
Run standalone:  python data/prepare_shakespeare.py
Or automatically when a run needs it:  trainer._ensure_data()
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from data.download import download_file, prepare_common, split_text  # noqa: E402
from src.utils.config import load_dataset_config  # noqa: E402
from src.utils.environment import data_dir  # noqa: E402

# Canonical tinyshakespeare mirror (karpathy/char-rnn). The old nanoGPT
# path (karpathy/nanoGPT/data/shakespeare_char/input.txt) is gone upstream;
# this URL serves the identical 1.1MB corpus.
URL = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"


def prepare(config: dict) -> dict:
    """Download + prepare; returns the trainer's data dict."""
    name = config["dataset"]
    ddir = os.path.join(data_dir(), name)
    raw_path = os.path.join(ddir, "input.txt")
    train_path = os.path.join(ddir, "train.txt")
    val_path = os.path.join(ddir, "val.txt")

    if not os.path.exists(raw_path):
        print(f"downloading Tiny Shakespeare -> {raw_path}")
        download_file(URL, raw_path, limit_chars=config.get("train_limit_chars"))

    if not (os.path.exists(train_path) and os.path.exists(val_path)):
        with open(raw_path, "r", encoding="utf-8", errors="replace") as f:
            text = f.read()
        train_text, val_text = split_text(text, config.get("val_ratio", 0.1))
        for path, part in ((train_path, train_text), (val_path, val_text)):
            with open(path, "w", encoding="utf-8") as f:
                f.write(part)

    return prepare_common(name, config, train_path, val_path)


if __name__ == "__main__":
    cfg = load_dataset_config("shakespeare")
    out = prepare(cfg)
    print("prepared:", {k: v for k, v in out.items() if "bin" in k})
