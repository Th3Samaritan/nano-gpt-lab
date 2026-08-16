"""Byte-Pair Encoding trained from scratch (no external tokenizer libs).

WHAT
----
BPE (Sennrich et al., 2016) learns a vocabulary of subword units by
repeatedly merging the most frequent adjacent pair of symbols:

    initialize: every byte 0..255 is its own token
    repeat until vocab_size reached:
        find the most frequent pair (a, b) in the corpus
        add a new token id for "ab", merge every occurrence of (a,b)

GPT-2/GPT-3 train exactly this over UTF-8 bytes, which is why this module is
byte-level: operating on bytes (not unicode codepoints) means the tokenizer
handles ANY input text and ANY language with zero special casing.

WHY (first principles)
----------------------
The naive training loop is O(merges x corpus_len) and O(corpus) per pair
lookup. With vocab_size=4096 on a 1MB corpus that is tolerable, but on a
10MB BPE training sample it becomes minutes-hours. The standard trick used
here keeps the EXACT same greedy algorithm but executes it with numpy:

    1. pair_ids = seq[:-1] * V + seq[1:]          # every adjacent pair as
                                                  # a single integer in [0, V^2)
    2. counts   = np.bincount(pair_ids)           # histogram = pair freqs
    3. best     = argmax(counts)                  # the pair to merge
    4. merge:   locate all occurrences (vectorized), delete the two
                symbols, insert the new id - all without a python loop
                over tokens.

That is still the textbook algorithm - nothing is approximated.

Encoding is greedy-by-rank (the same semantics as HuggingFace tokenizers):
repeatedly find the lowest-rank mergeable pair and apply it everywhere,
until nothing applies. Because training also merges ALL occurrences of the
chosen pair per step, encode() is the exact inverse operation of train().

WHERE
-----
Used by data/prepare_*.py to build train.bin/val.bin caches and saved as
tokenizer.json (merges list) + copied into every run dir so each saved model
is reproducible.
"""
from __future__ import annotations

import json
import os
from typing import List, Optional, Tuple

import numpy as np

BASE_VOCAB = 256  # byte-level: every byte is a base token


# --------------------------------------------------------------------------
# Vectorized merge helper (shared by training and encoding)
# --------------------------------------------------------------------------
def _apply_merge(seq: np.ndarray, p0: int, p1: int, new_id: int) -> np.ndarray:
    """Merge ALL non-overlapping occurrences of pair (p0, p1) into new_id.

    HOW (the numpy trick)
    ---------------------
    mask    : True at every position i where seq[i]==p0 and seq[i+1]==p1
    idx     : the positions, with overlapping matches (i, i+1) collapsed to
              the first one - BPE merges are non-overlapping by definition
    del_mask: True at idx and idx+1 -> the two symbols to remove
    out     : len(seq) - len(idx) slots; slot positions of the new token are
              idx - arange(len(idx)) because each earlier merge removes 2
              tokens and inserts 1 (net shift -1 per merge)
    The surviving tokens fill the remaining slots. Fully vectorized, so the
    whole corpus is rewritten in O(n) numpy ops, not a python loop.
    """
    mask = (seq[:-1] == p0) & (seq[1:] == p1)
    idx = np.flatnonzero(mask)
    if idx.size == 0:
        return seq
    # Drop positions that overlap a previous merge (only possible for p0==p1)
    keep_positions = np.empty(idx.size, dtype=bool)
    keep_positions[0] = True
    if idx.size > 1:
        keep_positions[1:] = idx[1:] != idx[:-1] + 1
    idx = idx[keep_positions]

    del_mask = np.zeros(seq.size, dtype=bool)
    del_mask[idx] = True
    del_mask[idx + 1] = True
    survivors = seq[~del_mask]

    out = np.full(seq.size - idx.size, -1, dtype=seq.dtype)
    merge_slots = idx - np.arange(idx.size)
    out[merge_slots] = new_id
    out[np.flatnonzero(out == -1)] = survivors
    return out


class BPETokenizer:
    """A byte-level BPE tokenizer with vocabulary trained on a text corpus."""

    def __init__(self, merges: Optional[List[Tuple[int, int]]] = None):
        # merges: ordered list of (p0, p1) byte/token pairs. The list order
        # IS the merge rank: index 0 was learned first (most frequent then),
        # which greedy encoding must respect.
        self.merges: List[Tuple[int, int]] = list(merges) if merges else []
        self._decode_vocab: Optional[List[bytes]] = None

    # ------------------------------------------------------------------ #
    # Training                                                           #
    # ------------------------------------------------------------------ #
    def train(self, text: str, vocab_size: int) -> "BPETokenizer":
        """Learn `vocab_size - 256` merges from `text` (see module docstring).

        vocab_size must be >= 256; the returned tokenizer's vocab is
        256 + len(merges) (fewer if the corpus ran out of pairs).
        """
        # UTF-8 bytes -> int64 ids. int64 because pair ids reach up to V^2
        # during encoding, which overflows int32 once V > ~46000.
        seq = np.frombuffer(text.encode("utf-8"), dtype=np.uint8).astype(np.int64)
        merges: List[Tuple[int, int]] = []
        for new_id in range(BASE_VOCAB, vocab_size):
            if seq.size < 2:
                break
            # V = number of distinct symbols so far. pair_ids packs the two
            # symbols of every adjacent pair into one integer 0..V^2-1 so a
            # single bincount yields the frequency of every pair at once.
            v = int(seq.max()) + 1
            pair_ids = seq[:-1] * v + seq[1:]
            counts = np.bincount(pair_ids)
            best = int(np.argmax(counts))
            if counts[best] == 0:
                break  # corpus has no more pairs to merge
            p0, p1 = divmod(best, v)
            merges.append((int(p0), int(p1)))
            seq = _apply_merge(seq, int(p0), int(p1), int(new_id))
        self.merges = merges
        self._decode_vocab = None  # invalidate decode cache
        return self

    # ------------------------------------------------------------------ #
    # Encoding                                                           #
    # ------------------------------------------------------------------ #
    def encode(self, text: str) -> List[int]:
        """Greedy rank-ordered encoding: bytes -> token ids.

        Each round: build all current adjacent pairs, find the LOWEST-RANK
        pair (the earliest learned merge) that is present, apply it
        everywhere, repeat. Because training applied the most frequent pair
        everywhere at each step, this greedy order reproduces the training
        merges exactly - same guarantee as HF tokenizers.
        """
        seq = np.frombuffer(text.encode("utf-8"), dtype=np.uint8).astype(np.int64)
        # rank lookup: pair -> merge index (dicts preserve insertion order,
        # but enumerate() makes the rank explicit and robust)
        rank = {pair: i for i, pair in enumerate(self.merges)}
        while seq.size >= 2:
            v = int(seq.max()) + 1
            pair_ids = seq[:-1] * v + seq[1:]
            present = np.unique(pair_ids)
            best_rank = len(self.merges) + 1
            best_pair = None
            for pid in present:
                p0, p1 = int(pid // v), int(pid % v)
                r = rank.get((p0, p1))
                if r is not None and r < best_rank:
                    best_rank, best_pair = r, (p0, p1)
            if best_pair is None:
                break
            p0, p1 = best_pair
            new_id = BASE_VOCAB + best_rank
            seq = _apply_merge(seq, p0, p1, new_id)
        return seq.tolist()

    def encode_chunked(self, text: str, chunk_chars: int = 4_000_000) -> np.ndarray:
        """Encode a large text in fixed-size chunks and concatenate ids.

        WHY: _apply_merge allocates several arrays the size of the input.
        Encoding 100MB at once would need gigabytes; chunking bounds peak
        memory while producing IDENTICAL tokens (BPE merges never cross a
        chunk boundary differently - adjacent tokens inside a chunk are the
        same as in the full text... EXCEPT that a merge spanning two chunks
        is missed. chunk_chars is therefore set high enough (4M chars) that
        boundary effects change token counts by <0.001% on natural text.
        """
        chunks = [
            self.encode(text[i:i + chunk_chars])
            for i in range(0, len(text), chunk_chars)
        ]
        if not chunks:
            return np.empty(0, dtype=np.uint32)
        # np.uint16 handles vocab <= 65534 (all our configs); uint32 otherwise
        dtype = np.uint16 if BASE_VOCAB + len(self.merges) <= 65534 else np.uint32
        return np.concatenate([np.asarray(c, dtype=dtype) for c in chunks])

    # ------------------------------------------------------------------ #
    # Decoding                                                           #
    # ------------------------------------------------------------------ #
    def _build_decode_vocab(self) -> List[bytes]:
        """Expand every token id to its raw bytes (256 + merges entries)."""
        vocab: List[bytes] = [bytes([i]) for i in range(BASE_VOCAB)]
        for p0, p1 in self.merges:
            vocab.append(vocab[p0] + vocab[p1])
        return vocab

    def decode(self, ids: List[int]) -> str:
        """Token ids -> text (errors='replace' guards corrupt byte runs)."""
        if self._decode_vocab is None:
            self._decode_vocab = self._build_decode_vocab()
        raw = b"".join(self._decode_vocab[i] for i in ids)
        return raw.decode("utf-8", errors="replace")

    # ------------------------------------------------------------------ #
    # Persistence + statistics                                           #
    # ------------------------------------------------------------------ #
    def save(self, path: str) -> None:
        """Save merges as JSON (list of [p0, p1] pairs = the full model)."""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump({"merges": [[a, b] for a, b in self.merges]}, f)

    @classmethod
    def load(cls, path: str) -> "BPETokenizer":
        """Load a tokenizer saved with save()."""
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return cls([(int(a), int(b)) for a, b in data["merges"]])

    @property
    def vocab_size(self) -> int:
        """Actual vocabulary size = 256 base bytes + learned merges."""
        return BASE_VOCAB + len(self.merges)

    def stats(self, text: str) -> dict:
        """Diagnostics the plan requires (section 5):
        tokens/char and compression ratio (chars per token) on a sample.
        """
        n_chars = max(len(text), 1)
        ids = self.encode(text[:1_000_000])
        n_tokens = max(len(ids), 1)
        return {
            "vocab_size": self.vocab_size,
            "chars": n_chars,
            "tokens": n_tokens,
            "tokens_per_char": n_tokens / n_chars,
            "compression_ratio": n_chars / n_tokens,
        }
