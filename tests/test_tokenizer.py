"""BPE tokenizer tests (plan section 29): roundtrip, compression,
vocab respect, determinism, persistence, unknown-input safety.

The roundtrip test is the tokenizer's contract with everything else:
prompt -> ids -> model -> ids -> text must be lossless, or generated
samples are silently corrupted.
"""
from src.tokenizer.bpe import BPETokenizer

import os  # noqa: E402  (used by test_save_load_roundtrip)

SAMPLE = (
    "the quick brown fox jumps over the lazy dog. "
    "The quick brown fox jumps over the lazy dog again and again. "
) * 10


def test_roundtrip():
    tok = BPETokenizer().train(SAMPLE, vocab_size=300)
    ids = tok.encode(SAMPLE)
    assert tok.decode(ids) == SAMPLE


def test_compression_on_repetition():
    """Frequent words must collapse into single tokens (the point of BPE)."""
    tok = BPETokenizer().train(SAMPLE, vocab_size=400)
    ids = tok.encode(SAMPLE)
    assert len(ids) < len(SAMPLE)  # fewer tokens than characters
    assert len(ids) * 1.0 / len(SAMPLE) < 0.8


def test_vocab_respected():
    tok = BPETokenizer().train(SAMPLE, vocab_size=300)
    ids = tok.encode(SAMPLE)
    assert max(ids) < 300
    assert tok.vocab_size <= 300


def test_deterministic_training():
    t1 = BPETokenizer().train(SAMPLE, vocab_size=256 + 50)
    t2 = BPETokenizer().train(SAMPLE, vocab_size=256 + 50)
    assert t1.merges == t2.merges


def test_identity_vocab():
    """vocab_size=256 -> no merges -> identity over bytes."""
    tok = BPETokenizer().train(SAMPLE, vocab_size=256)
    ids = tok.encode("abc")
    assert ids == [97, 98, 99]


def test_empty_and_tiny_inputs():
    tok = BPETokenizer().train(SAMPLE, vocab_size=300)
    assert tok.encode("") == []
    assert tok.decode([]) == ""
    assert tok.decode(tok.encode("a")) == "a"


def test_unicode_and_unknown_bytes():
    """Byte-level BPE must handle ANY unicode without special casing."""
    tok = BPETokenizer().train(SAMPLE, vocab_size=300)
    weird = "héllo wörld 中文 🚀 \u00e9\u0301"  # combining accent + emoji
    ids = tok.encode(weird)
    assert tok.decode(ids) == weird


def test_save_load_roundtrip():
    # tempfile (not pytest's tmp_path): this test must also run under
    # scripts/run_tests.py, which has no fixture injection.
    import tempfile
    with tempfile.TemporaryDirectory() as tmp:
        tok = BPETokenizer().train(SAMPLE, vocab_size=320)
        path = os.path.join(tmp, "tok.json")
        tok.save(path)
        tok2 = BPETokenizer.load(path)
        assert tok.merges == tok2.merges
        ids = tok.encode(SAMPLE)
        assert tok2.decode(ids) == SAMPLE


def test_chunked_matches_encode():
    """encode_chunked on a short text must equal encode exactly."""
    tok = BPETokenizer().train(SAMPLE, vocab_size=400)
    ids_a = tok.encode(SAMPLE)
    import numpy as np
    ids_b = tok.encode_chunked(SAMPLE, chunk_chars=10_000)
    assert np.array_equal(np.asarray(ids_a, dtype=np.uint16), ids_b)


def test_stats_plausible():
    tok = BPETokenizer().train(SAMPLE, vocab_size=400)
    stats = tok.stats(SAMPLE)
    assert stats["compression_ratio"] > 1.0
    assert stats["vocab_size"] <= 400
