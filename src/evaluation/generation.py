"""Standardized generation used by every experiment (plan section 27).

WHAT
----
generate_text() runs the model's sampler with FIXED parameters taken from
the dataset config's eval block (temperature / top_k / top_p /
max_new_tokens) on the SAME prompt for every variant. Samples are appended
to samples.txt inside the run dir with a header recording the step.

WHY
----
Hand-picking the best-looking sample is banned by the plan (section 27):
the comparison must be apples-to-apples, so every run writes samples at
the same intervals with the same prompt and sampler settings. The prompt
is short but tokenizer-encoded, so it also proves the BPE roundtrip
(prompt -> ids -> model -> ids -> text) works end to end.
"""
from __future__ import annotations

from typing import Optional

import torch

from src.tokenizer.bpe import BPETokenizer


def generate_text(
    model: torch.nn.Module,
    tokenizer: BPETokenizer,
    prompt: str,
    temperature: float = 0.8,
    top_k: Optional[int] = 40,
    top_p: Optional[float] = None,
    max_new_tokens: int = 200,
) -> str:
    """One sample with identical settings across experiments. Returns text.

    The model may be DataParallel-wrapped (multi-GPU runs): sampling runs
    on the base GPT so the wrapper's batch-splitting semantics never touch
    autoregressive generation (generation is strictly sequential).
    """
    from src.utils.device import get_base_gpt
    base = get_base_gpt(model)
    device = next(base.parameters()).device
    ids = torch.tensor([tokenizer.encode(prompt)], dtype=torch.long,
                       device=device)
    base.eval()
    out = base.generate(
        ids, max_new_tokens=max_new_tokens, temperature=temperature,
        top_k=top_k, top_p=top_p)
    base.train()
    return tokenizer.decode(out[0].tolist())


def load_tokenizer_for_run(tokenizer_path: str) -> BPETokenizer:
    """Load the BPE tokenizer a run was trained with."""
    return BPETokenizer.load(tokenizer_path)
