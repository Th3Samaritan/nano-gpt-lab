"""Dataset download + the shared BPE-tokenization-to-.bin pipeline.

WHAT
----
download_file() streams a URL to disk with a progress bar and an optional
character limit (used to take a documented SUBSET of big datasets - the
plan forbids silently training on "some of the data" without recording it,
so the limit lands in the dataset config and in meta.json).

prepare_common() is the shared tail of every dataset preparer:
    raw text -> train BPE from a sample -> tokenize -> train.bin/val.bin
                -> tokenizer.json + meta.json (vocab, token counts,
                tokens/char, compression ratio, subset limits)

WHY
----
All four experiment variants must share ONE tokenizer and ONE split; if
each preparer built its own, a tokenizer difference would contaminate the
A/B/C/D ablation. The .bin caches make training fast (memmap) and are
regenerated transparently when missing.

WHERE
-----
Called lazily by training/trainer.py (_ensure_data) so a run never blocks
on data it already has; also runnable standalone via each prepare_*.py.
"""
from __future__ import annotations

import json
import os
import shutil
from typing import Optional

from src.tokenizer.bpe import BPETokenizer
from src.utils.environment import data_dir


def download_file(
    url: str,
    dest: str,
    limit_chars: Optional[int] = None,
) -> None:
    """Stream url -> dest. limit_chars stops the download early (subsets).

    Retries once on failure - Colab/Kaggle connections occasionally drop
    mid-stream, and a truncated file must never look like a complete one
    (hence the .part extension until fully written).
    """
    import requests
    from tqdm import tqdm

    os.makedirs(os.path.dirname(dest), exist_ok=True)
    part = dest + ".part"
    for attempt in (1, 2):
        try:
            with requests.get(url, stream=True, timeout=60) as r:
                r.raise_for_status()
                total = int(r.headers.get("content-length") or 0)
                written = 0
                with open(part, "wb") as f, tqdm(
                        total=min(total, limit_chars or total) or None,
                        unit="B", unit_scale=True,
                        desc=os.path.basename(dest)) as pbar:
                    for chunk in r.iter_content(1024 * 1024):
                        if not chunk:
                            continue
                        if limit_chars is not None:
                            remaining = limit_chars - written
                            if remaining <= 0:
                                break
                            chunk = chunk[:remaining]
                        f.write(chunk)
                        written += len(chunk)
                        pbar.update(len(chunk))
            os.replace(part, dest)
            return
        except Exception as exc:
            print(f"download attempt {attempt} failed ({exc})")
            if attempt == 2:
                if os.path.exists(part):
                    os.remove(part)
                raise RuntimeError(f"could not download {url}")


def prepare_common(
    dataset_name: str,
    config: dict,
    train_text_path: str,
    val_text_path: str,
) -> dict:
    """BPE-train + tokenize a prepared raw text pair into .bin caches.

    Returns the dict the trainer consumes:
        train_bin, val_bin, tokenizer_path, meta_path, token_dtype
    Everything is cached: if all outputs exist, they are reused as-is.
    """
    ddir = os.path.join(data_dir(), dataset_name)
    os.makedirs(ddir, exist_ok=True)
    tokenizer_path = os.path.join(ddir, "tokenizer.json")
    train_bin = os.path.join(ddir, "train.bin")
    val_bin = os.path.join(ddir, "val.bin")
    meta_path = os.path.join(ddir, "meta.json")

    out = {"train_bin": train_bin, "val_bin": val_bin,
           "tokenizer_path": tokenizer_path, "meta_path": meta_path}

    if all(os.path.exists(p) for p in (train_bin, val_bin, tokenizer_path,
                                       meta_path)):
        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)
        out["token_dtype"] = meta.get("token_dtype", "uint16")
        return out

    # ---- 1. read the raw training text (and the BPE training sample) ----
    with open(train_text_path, "r", encoding="utf-8", errors="replace") as f:
        train_text = f.read()
    sample_chars = config.get("bpe_sample_chars")
    bpe_text = train_text if sample_chars is None else train_text[:sample_chars]

    # ---- 2. train BPE from scratch on the sample -------------------------
    vocab_size = config["vocab_size"]
    tokenizer = BPETokenizer().train(bpe_text, vocab_size)
    tokenizer.save(tokenizer_path)
    stats = tokenizer.stats(train_text[:1_000_000])

    # ---- 3. tokenize train + val into uint16/uint32 .bin caches ----------
    for text_path, bin_path, name in (
            (train_text_path, train_bin, "train"),
            (val_text_path, val_bin, "val")):
        if os.path.exists(bin_path):
            continue
        print(f"tokenizing {dataset_name}/{name}...")
        with open(text_path, "r", encoding="utf-8", errors="replace") as f:
            text = f.read()
        ids = tokenizer.encode_chunked(
            text, chunk_chars=config.get("encode_chunk_chars", 4_000_000))
        # Write via a temp file then rename: crash-safe cache writes.
        tmp = bin_path + ".tmp"
        with open(tmp, "wb") as f:
            f.write(ids.tobytes())
        os.replace(tmp, bin_path)

    # ---- 4. metadata (recorded so a subset is never silent) --------------
    train_n = os.path.getsize(train_bin) // 2
    val_n = os.path.getsize(val_bin) // 2
    token_dtype = "uint16" if tokenizer.vocab_size <= 65534 else "uint32"
    if token_dtype == "uint32":
        train_n //= 2
        val_n //= 2
    meta = {
        "dataset": dataset_name,
        "vocab_size": tokenizer.vocab_size,
        "train_tokens": train_n,
        "val_tokens": val_n,
        "token_dtype": token_dtype,
        "bpe_sample_chars": sample_chars,
        "stats": stats,
    }
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    out["token_dtype"] = token_dtype
    return out


def split_text(text: str, val_ratio: float) -> tuple:
    """Split text into (train, val) by characters (90/10 default).

    Character-level split keeps documents intact statistically without any
    tokenizer dependency - the split happens BEFORE BPE training so the
    tokenizer never sees validation text (no leakage into the vocab).
    """
    cut = int(len(text) * (1.0 - val_ratio))
    return text[:cut], text[cut:]
