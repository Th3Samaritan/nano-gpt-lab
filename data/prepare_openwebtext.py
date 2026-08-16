"""OpenWebText preparation (Dataset 3 - realistic pretraining subset).

WHAT
----
OpenWebText is the open re-creation of GPT-2's WebText (~40GB, 8M pages).
This preparer takes a DOCUMENTED SUBSET (plan section 21) via the
HuggingFace `datasets` streaming API - default 100MB of text - and feeds
it through the shared BPE pipeline.

Two ingestion paths:
    1. `datasets` installed (pip install datasets): stream
       Skylion007/openwebtext, concatenate the first N characters.
    2. A local text file: set "local_file" in configs/openwebtext.yaml
       to skip the HF dependency entirely.

WHY
----
The plan's Phase 7 question is whether small-data conclusions survive in a
realistic pretraining distribution. A fixed subset keeps it affordable on
T4 while staying faithful to web text statistics.

WHERE
-----
Colab/Kaggle tier only (needs the install + the bandwidth budget).
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from data.download import prepare_common  # noqa: E402
from src.utils.config import load_dataset_config  # noqa: E402
from src.utils.environment import data_dir  # noqa: E402


def _stream_hf_subset(limit_chars: int, out_path: str) -> None:
    """Stream the HF dataset until limit_chars characters are written."""
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise RuntimeError(
            "OpenWebText preparation needs the 'datasets' package "
            "(pip install datasets), or set 'local_file' in "
            "configs/openwebtext.yaml to use a text file you already have."
        ) from exc
    print(f"streaming Skylion007/openwebtext until {limit_chars:,} chars...")
    ds = load_dataset("Skylion007/openwebtext", split="train", streaming=True)
    with open(out_path, "w", encoding="utf-8") as f:
        written = 0
        for i, example in enumerate(ds):
            text = example.get("text", "")
            if not text:
                continue
            remaining = limit_chars - written
            if remaining <= 0:
                break
            chunk = text[:remaining]
            f.write(chunk)
            if len(chunk) > 0 and not chunk.endswith("\n"):
                f.write("\n")  # keep document boundaries clean
            written += len(chunk)
            if i % 1000 == 0:
                print(f"  {written:,}/{limit_chars:,} chars")
    print(f"  done: {written:,} chars")


def prepare(config: dict) -> dict:
    """Assemble the subset (HF stream or local file) + run shared pipeline."""
    name = config["dataset"]
    ddir = os.path.join(data_dir(), name)
    raw_path = os.path.join(ddir, "raw_subset.txt")
    train_path = os.path.join(ddir, "train.txt")
    val_path = os.path.join(ddir, "val.txt")

    if not os.path.exists(raw_path):
        local = config.get("local_file")
        if local and os.path.exists(local):
            print(f"using local OpenWebText file: {local}")
            with open(local, "r", encoding="utf-8", errors="replace") as src, \
                 open(raw_path, "w", encoding="utf-8") as dst:
                limit = config.get("train_limit_chars")
                dst.write(src.read(limit) if limit else src.read())
        else:
            _stream_hf_subset(
                config.get("train_limit_chars", 100_000_000), raw_path)

    if not (os.path.exists(train_path) and os.path.exists(val_path)):
        with open(raw_path, "r", encoding="utf-8", errors="replace") as f:
            text = f.read()
        from data.download import split_text
        train_text, val_text = split_text(text, config.get("val_ratio", 0.05))
        for path, part in ((train_path, train_text), (val_path, val_text)):
            with open(path, "w", encoding="utf-8") as f:
                f.write(part)

    return prepare_common(name, config, train_path, val_path)


if __name__ == "__main__":
    cfg = load_dataset_config("openwebtext")
    out = prepare(cfg)
    print("prepared:", {k: v for k, v in out.items() if "bin" in k})
