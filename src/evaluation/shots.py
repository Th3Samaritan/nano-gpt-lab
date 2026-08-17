"""Zero / one / few-shot (in-context learning) evaluation.

WHAT
----
In-context learning asks: given k example pairs shown INSIDE the prompt
(no weight updates), can the model continue the pattern on a new item?
This module implements the standard candidate-ranking protocol:

    build task from raw text:
        sentence pairs (context, next-sentence continuation) sampled from
        the dataset's own validation text - no external benchmark needed
    per test item with n_shots:
        prompt = shot_1..shot_k [context_1 COMPLETION_1] ... + new context
        candidates = true continuation + 3 wrong continuations (negatives
        drawn from OTHER stories/lines)
        the model scores each candidate by its negative log-likelihood
        given the prompt; picking the lowest-NLL candidate = correct

    accuracy(n_shots) is reported for n_shots in {0, 1, 3, 5} - the
    zero/one/few-shot curve.

WHY
----
This is the same protocol used to measure emergent ICL (Brown et al. 2020
style discrimination), rebuilt from first principles on any corpus. It
answers a question loss cannot: does the model actually USE the pattern
shown in context, and does more context help?

WHERE
-----
scripts/evaluate.py writes few_shot: {shots: accuracy} into
eval_report.json; the layman chart 05_few_shot.png plots the curve per
variant.
"""
from __future__ import annotations

import random
import re
from typing import List, Optional, Tuple

import torch
import torch.nn.functional as F

from src.tokenizer.bpe import BPETokenizer
from src.utils.device import get_device

_SENT_SPLIT = re.compile(r"(?<=[.!?])\s+")


def split_sentences(text: str, min_chars: int = 20) -> List[str]:
    """Rough sentence splitter - good enough to build (context, next)
    pairs from any corpus (TinyStories stories, Shakespeare lines)."""
    parts = _SENT_SPLIT.split(text)
    out = []
    for p in parts:
        p = p.strip()
        if len(p) >= min_chars:
            out.append(p)
    return out


def build_icl_task(text: str, n_items: int = 40, n_examples: int = 30,
                   seed: int = 0, min_chars: int = 20) -> Tuple[List[Tuple[str, str]], List[Tuple[str, str]]]:
    """Split text into (context, continuation) pairs; return
    (test_items, example_pool) drawn DETERMINISTICALLY (seed) so every
    variant evaluates on the same items and the same shot pool."""
    sents = split_sentences(text, min_chars=min_chars)
    if len(sents) < n_items + n_examples + 4:
        raise ValueError(
            f"not enough text for an ICL task ({len(sents)} sentences)")
    pairs = [(sents[i], sents[i + 1]) for i in range(len(sents) - 1)]
    rng = random.Random(seed)
    pool = list(pairs)
    rng.shuffle(pool)
    items = pool[:n_items]
    examples = pool[n_items:n_items + n_examples]
    return items, examples


def build_prompt(tokenizer: BPETokenizer, shots: List[Tuple[str, str]],
                 context: str) -> List[int]:
    """Token ids of the n-shot prompt.

    CRITICAL formatting rule: prompts must look like TRAINING text. The
    corpus was written 'sentence. sentence. sentence.' (period -> space ->
    next sentence), so shots and context are joined VERBATIM with spaces -
    never with newlines, which the model has never seen and which blank
    the ranking signal."""
    ids: List[int] = []
    for ctx, comp in shots:
        ids += tokenizer.encode(ctx)
        ids += tokenizer.encode(" " + comp + " ")
    ids += tokenizer.encode(context)
    return ids


def score_continuation(model, tokenizer: BPETokenizer,
                       prompt_ids: List[int], completion: str,
                       precision) -> float:
    """NLL of `completion` given the prompt (lower = more plausible).

    The completion is scored WITH its leading space ('period + space +
    next sentence' is the trained pattern); the space itself carries no
    discrimination (all candidates share it) but must be present or every
    candidate eats the same wrong-first-token penalty and the signal dies.

    One forward over prompt+completion; MEAN per-token cross-entropy over
    the completion span (the length-unbiased industry-standard form: a
    sum would penalize longer completions just for being longer)."""
    device = get_device()
    comp_ids = tokenizer.encode(" " + completion)
    if not comp_ids:
        return float("inf")
    seq = prompt_ids + comp_ids
    x = torch.tensor([seq], dtype=torch.long, device=device)
    model.eval()
    with torch.no_grad():
        with precision.autocast_ctx:
            out = model(x)
            logits = out[0] if isinstance(out, tuple) else out
        # targets are the shifted sequence; CE over the completion span
        targets = x[:, 1:]
        per_token = F.cross_entropy(
            logits[:, :-1].reshape(-1, logits.size(-1)),
            targets.reshape(-1), reduction="none")
        start = max(len(prompt_ids) - 1, 0)
        span = per_token[start:start + len(comp_ids)]
        nll = float(span.mean())
    model.train()
    return nll


def rank_accuracy(model, tokenizer: BPETokenizer,
                  items: List[Tuple[str, str]],
                  example_pool: List[Tuple[str, str]],
                  precision, n_shots: int = 0, n_negatives: int = 3,
                  seed: int = 0) -> float:
    """Candidate-ranking accuracy at n_shots (the ICL protocol).

    For each test item: 1 true continuation + n_negatives wrong ones from
    the pool; the model 'answers correctly' when the true continuation has
    the lowest NLL under the n-shot prompt."""
    rng = random.Random(seed + n_shots)
    correct = 0
    total = 0
    neg_completions = [c for _, c in example_pool]
    for ctx, true_comp in items:
        shots = rng.sample(example_pool, min(n_shots, len(example_pool)))
        prompt_ids = build_prompt(tokenizer, shots, ctx)
        candidates = [true_comp] + rng.sample(neg_completions, n_negatives)
        nlls = [score_continuation(model, tokenizer, prompt_ids, c, precision)
                for c in candidates]
        if nlls.index(min(nlls)) == 0:
            correct += 1
        total += 1
    return correct / max(total, 1)
