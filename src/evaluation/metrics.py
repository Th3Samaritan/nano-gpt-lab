"""Industry-standard evaluation metrics beyond loss/perplexity.

WHAT
----
Loss and perplexity measure average next-token surprise. The industry asks
more of a language model, so this module adds the standard battery:

    bits_per_byte          - compression efficiency (tokens -> bytes),
                             the classic "how much information per byte"
                             LM metric (GPT-2's bpb, used in the paper)
    expected_calibration_error (ECE)
                           - does the model KNOW what it knows? 80% confident
                             predictions should be right ~80% of the time.
                             15-bin reliability statistic (Guo et al. 2017)
    distinct_n             - generation diversity: fraction of unique
                             n-grams in generated text (Li et al. 2016)
    repetition_rate        - fraction of repeated consecutive tokens
    loss_by_position_bucket- long-range penalty: is later-context prediction
                             worse than early-context? (4 buckets)
    inference_tokens_per_sec - deployment-facing speed (generation tokens/s)

WHY
----
'Lower val loss' can hide a model that is miscalibrated, repetitive, or
collapses at long context. These metrics are the layman-readable answers
to 'can I trust it, is it varied, and how fast is it in production?'

WHERE
-----
Used by scripts/evaluate.py (writes eval_report.json per run) and rendered
by the 05-08 layman charts in src/utils/plots.py.
"""
from __future__ import annotations

import math
import time

import torch
import torch.nn.functional as F

from src.utils.device import get_base_gpt, get_device


def bits_per_byte(loss: float, tokens_per_char: float) -> float:
    """Compression efficiency: loss is nats/token; convert to bits per
    character (byte-level BPE => chars == bytes here).

    bpb = loss / ln(2) * tokens_per_char
    GPT-2 (WebText) reports ~1.0 bpb for reference.
    """
    return loss / math.log(2) * tokens_per_char


def collect_confidence_samples(model, get_batch, precision, n_batches: int = 10):
    """Gather (confidence, correctness) pairs for ECE over random batches.

    confidence = the model's top-1 probability; correctness = whether that
    top-1 token is the true next token."""
    device = get_device()
    model.eval()
    confs, corrects = [], []
    with torch.no_grad():
        for _ in range(n_batches):
            x, y = get_batch()
            x, y = x.to(device), y.to(device)
            with precision.autocast_ctx:
                out = model(x)
                logits = out[0] if isinstance(out, tuple) else out
            probs = F.softmax(logits, dim=-1)
            conf, pred = probs.max(dim=-1)
            confs.append(conf.view(-1).float().cpu())
            corrects.append((pred.view(-1) == y.view(-1).to(pred.device))
                            .float().cpu())
    model.train()
    return torch.cat(confs), torch.cat(corrects)


def expected_calibration_error(confs: torch.Tensor, corrects: torch.Tensor,
                               n_bins: int = 15) -> float:
    """ECE = weighted average of |accuracy(bin) - confidence(bin)|.

    Perfect calibration => 0. A model whose confident guesses are wrong
    gets a large ECE (e.g. always 90% confident but 60% right => ~0.3).

    Bins are HALF-OPEN [lo, hi): a boundary value must belong to exactly
    ONE bin, or it double-counts and inflates the weight of that bin."""
    ece = 0.0
    total = len(confs)
    for i in range(n_bins):
        lo, hi = i / n_bins, (i + 1) / n_bins
        if i == n_bins - 1:
            mask = (confs >= lo) & (confs <= 1.0)
        else:
            mask = (confs >= lo) & (confs < hi)
        n = int(mask.sum())
        if n == 0:
            continue
        bin_acc = float(corrects[mask].mean())
        bin_conf = float(confs[mask].mean())
        ece += (n / total) * abs(bin_acc - bin_conf)
    return float(ece)


def distinct_n(tokens, n: int) -> float:
    """Fraction of unique n-grams among generated tokens (diversity).

    distinct-1 = unique words / total words; distinct-2 does bigrams.
    A model stuck in a loop collapses toward 0."""
    if len(tokens) < n:
        return 0.0
    ngrams = {tuple(tokens[i:i + n]) for i in range(len(tokens) - n + 1)}
    return len(ngrams) / (len(tokens) - n + 1)


def repetition_rate(tokens) -> float:
    """Fraction of tokens that repeat the previous token (loop detector)."""
    if len(tokens) < 2:
        return 0.0
    return sum(1 for a, b in zip(tokens, tokens[1:]) if a == b) / len(tokens)


def loss_by_position_bucket(model, get_batch, precision, n_buckets: int = 4,
                            n_batches: int = 3):
    """Mean next-token loss per CONTEXT-POSITION bucket.

    Bucket 0 = predictions early in the window, bucket 3 = at the far end.
    The early/late gap is the 'long-range penalty': architectures that
    cannot use distant context (or that RoPE helps) show it directly."""
    device = get_device()
    model.eval()
    bucket_sums = torch.zeros(n_buckets, device="cpu")
    bucket_counts = torch.zeros(n_buckets, device="cpu")
    with torch.no_grad():
        for _ in range(n_batches):
            x, y = get_batch()
            x, y = x.to(device), y.to(device)
            with precision.autocast_ctx:
                out = model(x)
                logits = out[0] if isinstance(out, tuple) else out
            T = x.size(1)
            per_token = F.cross_entropy(
                logits.view(-1, logits.size(-1)), y.view(-1),
                reduction="none").view(x.size(0), T).cpu()
            bucket_id = (torch.arange(T) * n_buckets // T).cpu()
            for b in range(n_buckets):
                mask = bucket_id == b
                bucket_sums[b] += per_token[:, mask].sum()
                bucket_counts[b] += int(mask.sum()) * x.size(0)
    model.train()
    return [float(bucket_sums[b] / max(bucket_counts[b], 1))
            for b in range(n_buckets)]


def inference_tokens_per_sec(model, tokenizer, prompt: str,
                             n_tokens: int = 100) -> float:
    """Generation (deployment-facing) speed in tokens/second."""
    from src.evaluation.generation import generate_text
    device = next(get_base_gpt(model).parameters()).device
    t0 = time.time()
    generate_text(model, tokenizer, prompt, max_new_tokens=n_tokens,
                  temperature=0.8, top_k=40)
    return n_tokens / max(time.time() - t0, 1e-6)
