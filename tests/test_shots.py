"""Few-shot evaluation tests: the ICL task machinery must be sound.

The end-to-end test trains a tiny model on a corpus of DISTINCT topics and
checks the zero-shot protocol picks the true continuation over negatives -
proving the task builder, prompt builder, and NLL scorer all work (and
that a memorized model passes an ICL discrimination test, as it should).
"""
import torch

from src.evaluation.shots import (
    build_icl_task, build_prompt, rank_accuracy, score_continuation, split_sentences)
from src.model.gpt import GPT, GPTConfig
from src.tokenizer.bpe import BPETokenizer
from src.training.optimizer import create_optimizer
from src.training.precision import PrecisionContext
from src.training.scheduler import CosineWarmupScheduler
from src.utils.device import get_autocast_dtype, get_device
from src.utils.seed import seed_all


def test_split_and_task_builder():
    sents_text = " ".join(
        f"topic number {i} starts right here now. topic number {i} ends here."
        for i in range(12))
    sents = split_sentences(sents_text)
    assert len(sents) == 24
    items, examples = build_icl_task(sents_text, n_items=4, n_examples=4)
    assert len(items) == 4 and len(examples) == 4
    ctx, comp = items[0]
    assert len(ctx) > 0 and len(comp) > 0


def test_prompt_builder_contains_everything():
    tok = BPETokenizer()  # identity (256) - no merges
    # Verbatim spacing (period -> space -> next sentence), never newlines:
    prompt = build_prompt(tok, [("alpha.", "beta.")], "gamma.")
    decoded = tok.decode(prompt)
    assert decoded == "alpha. beta. gamma."


def test_icl_end_to_end_zero_shot():
    """Tiny trained model must rank the true continuation above wrong ones.

    Design notes (learned the hard way):
    - only 3 topics, so the tiny model is CERTAIN to memorize the mapping;
    - pairs are built DIRECTLY (start -> its own ending), not from the
      cyclic corpus, so every context has an unambiguous continuation;
    - completions differ across MANY tokens (name + unique phrase), so
      the ranking signal is strong.
    """
    seed_all(0)
    device = get_device()
    names = ["alpha", "bravo", "charlie"]
    text = " ".join(
        f"the story of {n} begins here. the ending of {n} is unique-{n}-one."
        for n in names)
    text = (text + " ") * 10
    tokenizer = BPETokenizer().train(text, vocab_size=256)  # identity bytes
    import numpy as np
    ids = tokenizer.encode_chunked(text)

    # block_size 128: byte-level identity tokenizer means 1 char = 1 token,
    # so prompt+completion (~80 tokens) must fit for the ICL scorer.
    cfg = GPTConfig(vocab_size=256, block_size=128, n_layer=2, n_head=2,
                    n_embd=32, attention="vanilla", position="learned")
    model = GPT(cfg).to(device)
    optimizer = create_optimizer(model, weight_decay=0.0, learning_rate=2e-3,
                                 betas=(0.9, 0.95), device=device)
    scheduler = CosineWarmupScheduler(optimizer, warmup_steps=10,
                                      decay_steps=590, min_lr=3e-4)
    precision = PrecisionContext(device, get_autocast_dtype(device))
    B, T = 16, 64
    data = torch.from_numpy(ids.astype(np.int64))
    for step in range(600):
        ix = torch.randint(0, len(data) - T - 1, (B,))
        x = torch.stack([data[i:i + T] for i in ix]).to(device)
        y = torch.stack([data[i + 1:i + 1 + T] for i in ix]).to(device)
        optimizer.zero_grad(set_to_none=True)
        with precision.autocast_ctx:
            _, loss = model(x, y)
        precision.backward(loss)
        precision.optimizer_step(optimizer, grad_clip=1.0)
        scheduler.step()

    items = [(f"the story of {n} begins here.",
              f"the ending of {n} is unique-{n}-one.") for n in names]
    acc = rank_accuracy(model, tokenizer, items, items, precision,
                        n_shots=0)
    assert acc >= 0.85, f"zero-shot ICL accuracy too low: {acc}"
    # And the scorer itself: the true continuation must score lower than a
    # continuation of a DIFFERENT topic.
    ctx, true_comp = items[0]
    other = items[1][1]
    prompt_ids = build_prompt(tokenizer, [], ctx)
    nll_true = score_continuation(model, tokenizer, prompt_ids, true_comp,
                                  precision)
    nll_other = score_continuation(model, tokenizer, prompt_ids, other,
                                   precision)
    assert nll_true < nll_other
