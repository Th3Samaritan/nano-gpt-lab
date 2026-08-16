"""Learning-rate schedule: linear warmup + cosine decay (GPT-3 recipe).

WHAT
----
lr(step):
    step < warmup_steps : lr = max_lr * step / warmup_steps
    else                : cosine from max_lr down to min_lr over the
                          remaining steps

WHY
----
Warmup prevents giant early gradients from destabilizing Adam's moment
estimates; cosine decay anneals into a low-loss basin instead of slamming
into the minimum-LR floor at the end. This schedule is applied identically
to every experiment - again so it cannot explain any A/B/C/D difference.
The scheduler is PURE (a function of step with no internal state), which
makes resume trivial: save the step, recompute lr from it.

WHERE
-----
Stepped once per optimizer step inside training/trainer.py; its state_dict
(just the step) is part of every checkpoint per the plan section 26.
"""
from __future__ import annotations

import math


class CosineWarmupScheduler:
    """Linear warmup then cosine decay (stateless - see module docstring)."""

    def __init__(self, optimizer, warmup_steps: int, decay_steps: int,
                 min_lr: float):
        self.optimizer = optimizer
        self.warmup_steps = max(warmup_steps, 0)
        self.decay_steps = max(decay_steps, 1)
        self.min_lr = min_lr
        # The schedule's peak lr is captured ONCE at construction. Reading
        # it back from the optimizer every step would be self-referential
        # (the optimizer holds the DECAYED value), collapsing lr
        # exponentially - a classic silent training killer.
        self.max_lr = optimizer.param_groups[0]["lr"]
        self.step_count = 0
        self.current_lr = 0.0

    def get_lr(self, step: int) -> float:
        """Learning rate at absolute step (the whole schedule formula)."""
        if step < self.warmup_steps:
            # Linear ramp from 0 to max_lr
            return self.max_lr * (step + 1) / max(self.warmup_steps, 1)
        if step >= self.warmup_steps + self.decay_steps:
            return self.min_lr
        # Cosine decay: 1.0 at warmup end, 0.0 at schedule end, mapped to
        # [min_lr, max_lr]
        progress = (step - self.warmup_steps) / self.decay_steps
        decayed = 0.5 * (1.0 + math.cos(math.pi * progress))
        return self.min_lr + (self.max_lr - self.min_lr) * decayed

    def step(self) -> float:
        """Advance one step, apply the new lr, return it."""
        self.current_lr = self.get_lr(self.step_count)
        self.step_count += 1
        for group in self.optimizer.param_groups:
            group["lr"] = self.current_lr
        return self.current_lr

    def state_dict(self) -> dict:
        return {"step_count": self.step_count}

    def load_state_dict(self, state: dict) -> None:
        self.step_count = state.get("step_count", 0)
