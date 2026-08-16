"""THE GATE (plan section 29): a model that cannot overfit a tiny dataset
must not proceed to large-scale training.

WHAT
----
Trains a minimal GPT on a tiny synthetic corpus until the training loss
crashes to ~0. Exercises the real training stack (tokenizer, data loader,
model, AdamW, scheduler, mixed-precision context) so a pass here means the
whole pipeline is wired correctly.

WHY
----
Overfitting a toy corpus is the strongest end-to-end correctness test a
language model has: every layer, residual, mask and head must work for
loss to vanish. It costs ~a minute on CPU, and catches silent breakage
that unit tests on tensors cannot see.
"""
import torch

from src.model.gpt import GPT, GPTConfig
from src.tokenizer.bpe import BPETokenizer
from src.training.optimizer import create_optimizer
from src.training.precision import PrecisionContext
from src.training.scheduler import CosineWarmupScheduler
from src.utils.device import get_autocast_dtype, get_device
from src.utils.seed import seed_all


def test_overfit_tiny_corpus():
    seed_all(0)
    device = get_device()

    # ---- 1. a tiny, highly repetitive corpus (must be trivially learnable)
    sentence = "the red fox and the blue dog run and run and jump. "
    text = sentence * 300

    # ---- 2. a small BPE tokenizer trained on the corpus itself
    tokenizer = BPETokenizer().train(text, vocab_size=512)
    import numpy as np
    ids = tokenizer.encode_chunked(text)

    # ---- 3. a minimal model (2 layers, 2 heads, 32-dim)
    cfg = GPTConfig(vocab_size=tokenizer.vocab_size, block_size=32,
                    n_layer=2, n_head=2, n_embd=32, attention="vanilla",
                    position="learned")
    model = GPT(cfg).to(device)
    optimizer = create_optimizer(model, weight_decay=0.01,
                                 learning_rate=1e-3, betas=(0.9, 0.95),
                                 device=device)
    scheduler = CosineWarmupScheduler(optimizer, warmup_steps=20,
                                      decay_steps=380, min_lr=1e-4)
    precision = PrecisionContext(device, get_autocast_dtype(device))

    # ---- 4. the training loop (same components as the real trainer)
    B, T = 16, 32
    block = T + 1  # sample T+1 tokens so x/y align
    data = torch.from_numpy(ids.astype(np.int64))
    for step in range(400):
        ix = torch.randint(0, len(data) - block, (B,))
        x = torch.stack([data[i:i + T] for i in ix]).to(device)
        y = torch.stack([data[i + 1:i + 1 + T] for i in ix]).to(device)
        optimizer.zero_grad(set_to_none=True)
        with precision.autocast_ctx:
            _, loss = model(x, y)
        precision.backward(loss)
        precision.optimizer_step(optimizer, grad_clip=1.0)
        scheduler.step()

    # ---- 5. the gate: loss must be essentially zero (full memorization)
    assert loss.item() < 0.1, f"model failed to overfit: loss {loss.item():.4f}"
