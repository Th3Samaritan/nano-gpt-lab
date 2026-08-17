# Transformer Ablation Laboratory — Extended Implementation Plan

## 1. Purpose

Extend the NanoGPT/GPT-2 first-principles project into a controlled experimental laboratory for studying how individual Transformer design choices affect:

- Language-model quality
- Convergence
- Training stability
- Training throughput
- GPU memory usage
- Parameter count
- Inference efficiency
- Compute efficiency

The project investigates three major categories:

1. Attention + positional representation
2. Activation function / feed-forward architecture
3. Optimizer and training dynamics

Core rule:

> Understand → implement → verify → establish a baseline → change one variable → benchmark → explain → combine only after isolated effects are understood.

---

## 2. Experimental Structure

Do not test every possible combination immediately.

### Study A — Attention × Position

| Experiment | Attention | Position |
|---|---|---|
| A1 | Vanilla causal attention | Learned absolute |
| A2 | Vanilla causal attention | RoPE |
| A3 | Flash/fused attention | Learned absolute |
| A4 | Flash/fused attention | RoPE |

### Study B — Activation / FFN

Hold attention, position, optimizer, and training settings fixed.

| Experiment | Activation / FFN |
|---|---|
| B1 | GELU |
| B2 | ReLU |
| B3 | SiLU / Swish |
| B4 | SwiGLU |

### Study C — Optimizer

Hold architecture and dataset fixed.

| Experiment | Optimizer |
|---|---|
| C1 | Adam |
| C2 | AdamW |
| C3 | Future research candidate |

Only add a third optimizer when there is a concrete research reason.

### Study D — Interaction

Only after A–C are understood, combine selected components.

Example:

```text
RoPE
+
Flash/Fused Attention
+
SwiGLU
+
AdamW
```

The purpose is to discover whether gains combine, cancel, or interact.

---

## 3. Canonical Baseline

Use one fixed baseline:

```text
Decoder-only Transformer
+
BPE tokenizer
+
Learned absolute positional embeddings
+
Causal multi-head self-attention
+
GELU MLP
+
LayerNorm
+
AdamW
```

This is the control condition.

All experiments must clearly state which variable differs from it.

---

## 4. Modular Model Design

The codebase should make major components replaceable.

```text
Model
│
├── Token Embedding
│
├── Positional Representation
│   ├── Learned Absolute
│   └── RoPE
│
├── Transformer Blocks
│   ├── Attention
│   │   ├── Vanilla
│   │   └── Flash/Fused
│   ├── Normalization
│   │   └── LayerNorm initially
│   └── Feed-Forward
│       ├── GELU
│       ├── ReLU
│       ├── SiLU
│       └── SwiGLU
│
└── Language Model Head

Training
│
├── Optimizer
│   ├── Adam
│   ├── AdamW
│   └── Future candidates
├── Scheduler
├── Mixed Precision
└── Gradient Accumulation
```

No architecture experiment should require rewriting the training loop.

---

## 5. Activation Study

### Why?

The Transformer feed-forward network provides nonlinear transformation:

\[
FFN(x)=W_2 GELU(W_1x+b_1)+b_2
\]

Changing the activation can affect:

- nonlinear behavior
- gradients
- optimization
- compute
- final quality

### Sequence

**B0 — GELU baseline**

Implement:

```text
Linear
↓
GELU
↓
Linear
```

Verify shapes and gradients.

**B1 — ReLU**

Change only:

```text
GELU → ReLU
```

**B2 — SiLU / Swish**

Change only:

```text
GELU → SiLU
```

**B3 — SwiGLU**

Treat SwiGLU as an FFN architectural change, not simply a drop-in scalar activation.

Conceptually:

\[
SwiGLU(x)=SiLU(xW_g)\odot(xW_u)
\]

followed by the output projection.

### Important fairness rule

For SwiGLU, run two interpretations:

1. **Parameter-matched:** adjust hidden width to keep parameter counts approximately equal.
2. **Architecture-native:** use the conventional hidden width.

Report both clearly because they answer different questions.

### Metrics

Record:

- train loss
- validation loss
- perplexity
- tokens/sec
- step time
- peak GPU memory
- parameter count
- tokens to target loss
- total wall-clock time
- training stability

---

## 6. Optimizer Study

### Why?

The optimizer controls how gradients become parameter updates.

Basic gradient descent:

\[
	heta_{t+1}=	heta_t-\eta
abla_	heta L
\]

Modern optimizers add state and update rules.

### C1 — Adam

Understand:

- first moment
- second moment
- bias correction
- learning rate
- epsilon

### C2 — AdamW

Study decoupled weight decay separately from the gradient update.

Do not treat Adam and AdamW as identical.

### C3 — Future candidate

Only add a third optimizer when there is a documented motivation.

Potential future candidates may include research-oriented optimizers, but they should not be added merely to increase the number of experiments.

Each candidate must have:

- motivation
- implementation
- hyperparameter definition
- comparison protocol

---

## 7. Optimizer Fairness

Optimizer comparisons are sensitive to hyperparameters.

Use two levels:

### Primary comparison

Use a principled shared training configuration.

### Secondary comparison

Allow a documented small hyperparameter search.

Do not accidentally compare a well-tuned optimizer against a poorly tuned one.

Always record:

- learning rate
- weight decay
- scheduler
- warmup
- betas/moments
- epsilon
- gradient clipping

---

## 8. First-Principles Optimizer Work

Before large experiments:

1. Understand the mathematical update.
2. Implement a simple reference version when practical.
3. Compare against PyTorch.
4. Verify numerical behavior.
5. Verify optimizer state.
6. Test on a tiny model.

The production implementation may remain PyTorch-native; the point is understanding.

---

## 9. Attention + Position Study

Use the previously defined 2 × 2 matrix.

### A1

```text
Vanilla attention
+
Learned absolute position
```

### A2

```text
Vanilla attention
+
RoPE
```

### A3

```text
Flash/Fused attention
+
Learned absolute position
```

### A4

```text
Flash/Fused attention
+
RoPE
```

Keep the rest identical.

The purpose is to separate:

- effects from positional representation
- effects from attention implementation

Remember:

> FlashAttention primarily targets computational and memory efficiency.

Do not assume it should lower validation loss.

---

## 10. Research Questions

### Attention

Does fused attention change model quality, or mainly efficiency?

### Position

Does RoPE change validation quality or convergence, and does its effect depend on context length?

### Activation

Under comparable parameter counts, does SwiGLU provide a measurable advantage over GELU?

### Optimizer

Under a fixed training-token budget, does AdamW reach a target loss faster or achieve better final validation quality than Adam?

### Interaction

Do RoPE, FlashAttention, and SwiGLU provide additive gains, or do some gains disappear when combined?

---

## 11. Datasets

Run experiments progressively.

### Tiny Shakespeare

Use for:

- debugging
- rapid iteration
- overfit tests
- initial ablations

### TinyStories

Use for:

- more natural language
- transfer of conclusions from Shakespeare
- broader generation behavior

### OpenWebText

Use a fixed subset first.

Only increase dataset size after the pipeline is stable.

---

## 12. Tokenization

Use BPE consistently across the main comparisons.

Record:

- vocabulary size
- special tokens
- encoding/decoding policy
- average tokens per document
- sequence packing policy

Do not change tokenizers between A/B/C/D.

---

## 13. Compute Budget

Choose a fixed budget for controlled studies.

Useful budget units:

- training tokens
- optimizer steps
- wall-clock GPU time
- estimated FLOPs

For primary scientific comparisons, training tokens are usually the clearest baseline.

Always report the budget used.

---

## 14. Metrics

Separate three categories.

### Model quality

- validation loss
- perplexity
- generation samples

### Training efficiency

- tokens/sec
- step latency
- wall-clock time
- estimated FLOPs
- MFU when practical

### Memory

- parameter memory
- gradient memory
- optimizer state memory
- activation memory
- peak VRAM
- checkpoint size

---

## 15. Parameter Accounting

Every experiment should report:

```text
total parameters
trainable parameters
embedding parameters
attention parameters
FFN parameters
LM-head parameters
```

For LoRA/PEFT later, also report adapter parameters.

For SwiGLU, explicitly state whether a comparison is parameter-matched.

---

## 16. Unit Tests

Before serious training, test:

### Attention

- tensor shapes
- causal masking
- no future-token leakage

### RoPE

- frequency construction
- dimension pairing
- rotation
- broadcasting
- multiple sequence lengths

### FFN

- output shapes
- gradients
- parameter counts

### Optimizers

- state creation
- update correctness
- numerical stability

### Tokenizer

- encode/decode consistency
- special-token handling

### Training loop

- loss decreases on a tiny batch
- model can overfit a tiny dataset

Critical test:

> The model should be able to nearly memorize a tiny dataset.

If it cannot, do not move to large training.

---

## 17. Numerical Reference Testing

Once vanilla attention is implemented:

1. Create identical Q/K/V tensors.
2. Run your implementation.
3. Run the PyTorch reference.
4. Compare outputs.
5. Compare gradients.
6. Measure numerical error.

Do the same for RoPE and other first-principles components.

Only after validation should you benchmark large runs.

---

## 18. Configuration-Driven Experiments

Use YAML/TOML/JSON configurations.

Example:

```yaml
model:
  attention: vanilla
  position: rope
  activation: swiglu

training:
  optimizer: adamw
  learning_rate: 0.0003
  weight_decay: 0.1

data:
  dataset: shakespeare
  tokenizer: bpe
```

Another experiment changes only configuration:

```yaml
model:
  attention: flash
  position: learned
  activation: gelu

training:
  optimizer: adam
```

The Python training code should remain unchanged.

---

## 19. Run Organization

Use:

```text
runs/
├── attention_position/
│   ├── A1_vanilla_learned/
│   ├── A2_vanilla_rope/
│   ├── A3_flash_learned/
│   └── A4_flash_rope/
│
├── activation/
│   ├── B1_gelu/
│   ├── B2_relu/
│   ├── B3_silu/
│   └── B4_swiglu/
│
└── optimizer/
    ├── C1_adam/
    ├── C2_adamw/
    └── C3_future/
```

Suggested run name:

```text
{dataset}_{attention}_{position}_{activation}_{optimizer}_{seed}
```

---

## 20. Reproducibility

Every run should save:

```text
run_id
timestamp
git_commit
dataset
tokenizer
attention
position
activation
optimizer
learning_rate
weight_decay
batch_size
gradient_accumulation
context_length
parameter_count
trainable_parameter_count
dtype
seed
GPU
PyTorch version
training tokens
```

A result without its configuration is not a reproducible experiment.

---

## 21. Learning Curves

Automatically generate:

### Training

```text
training loss vs tokens
validation loss vs tokens
```

### Efficiency

```text
tokens/sec
step latency
peak VRAM
```

### Quality vs cost

```text
validation loss vs training tokens
validation loss vs wall-clock time
validation loss vs estimated FLOPs
```

For repeated seeds, show mean ± standard deviation.

---

## 22. Generation Evaluation

For comparable experiments, use identical:

- prompts
- temperature
- top-k
- top-p
- max-new-token count

Store generated samples with the experiment.

Generation samples are qualitative evidence, not a replacement for validation metrics.

Do not cherry-pick outputs.

---

## 23. Experimental Sequence

### Phase 0 — Environment

- PyTorch
- CUDA detection
- Colab/Kaggle detection
- local development
- reproducibility

### Phase 1 — First principles

- attention from scratch
- learned position
- RoPE
- GELU
- ReLU
- SiLU
- SwiGLU
- Adam
- AdamW

### Phase 2 — Tiny overfit

Verify every component on tiny data.

### Phase 3 — Shakespeare

Run the full attention/position matrix.

Then activation study.

Then optimizer study.

### Phase 4 — TinyStories

Repeat the important comparisons.

### Phase 5 — OpenWebText subset

Test transfer to a more realistic dataset.

### Phase 6 — GPT-2 class

Scale toward ~124M parameters.

### Phase 7 — Interactions

Combine the strongest/most interesting components.

### Phase 8 — New hypothesis

Use the evidence to formulate an original architectural or optimization question.

---

## 24. Local + Colab + Kaggle Workflow

### Local PC

Use the local machine for:

- first-principles coding
- debugging
- unit tests
- tiny models
- tokenizer work
- tiny overfit experiments
- architecture verification

### Colab / Kaggle

Use GPU environments for:

- larger NanoGPT
- TinyStories
- OpenWebText subsets
- FlashAttention benchmarks
- GPT-2-class experiments

Keep one codebase.

The environment should be detected automatically.

---

## 25. Checkpointing

A checkpoint should include:

```python
{
    "model_state_dict": ...,
    "optimizer_state_dict": ...,
    "scheduler_state_dict": ...,
    "scaler_state_dict": ...,
    "step": ...,
    "tokens_seen": ...,
    "config": ...,
    "rng_state": ...,
}
```

Maintain:

```text
latest.pt
best.pt
```

Resume interrupted experiments instead of restarting.

---

## 26. Colab Support

On Colab:

- mount Google Drive
- keep active training I/O local when practical
- periodically copy checkpoints/results to Drive
- resume from Drive after runtime termination

Example:

```text
/content/drive/MyDrive/nano-gpt-lab/
```

Save:

- checkpoints
- configs
- logs
- metrics
- generated samples
- plots

---

## 27. Kaggle Support

Use:

```text
/kaggle/working/
```

for active outputs.

Use Kaggle's input/data mechanisms for reusable datasets where practical.

The model and training code should not need to change because the environment changes.

---

## 28. NanoGPT Starting Point

Start smaller than GPT-2.

Example:

```yaml
n_layer: 4
n_head: 4
n_embd: 256
block_size: 256
dropout: 0.0
```

The exact values are not sacred.

The purpose is fast iteration.

Once the system is verified, scale upward.

---

## 29. GPT-2 Scaling

After the laboratory is stable, move toward approximately:

```text
~124M parameters
12 layers
12 attention heads
~768 hidden dimension
~1024 context
```

Do not attempt to recreate the original GPT-2 training compute on a single T4.

Instead:

1. Match the architecture.
2. Verify parameter count.
3. Verify forward/backward correctness.
4. Benchmark memory.
5. Train a manageable token budget.
6. Re-run the most informative ablations.

---

## 30. Future PEFT / LoRA Study

PEFT and LoRA should be introduced as a later **adaptation study**, not mixed into the initial pretraining ablations.

Study:

```text
Full fine-tuning
vs
PEFT
vs
LoRA
```

For LoRA, investigate:

- rank \(r\)
- target modules
- scaling \(lpha\)
- adapter parameter count
- memory
- training time
- downstream performance

Conceptually:

\[
W' = W + rac{lpha}{r}BA
\]

where:

\[
A\in\mathbb{R}^{r	imes d}
\]

and:

\[
B\in\mathbb{R}^{d	imes r}
\]

with:

\[
r\ll d
\]

This should be studied after the base Transformer and optimizer behavior are understood.

---

## 31. First-Principles "Naked" Implementation Rule

For every important component:

1. Implement the mathematical version yourself.
2. Write tests.
3. Compare to the optimized/library version.
4. Benchmark both.
5. Record the trade-off.

The optimized implementation should be treated as something to understand, not a black box.

---

## 32. Research Notebook Template

Every experiment should answer:

```text
Hypothesis

Why should this happen?

What exactly changed?

What stayed fixed?

Expected result

Observed result

Unexpected result

Possible explanation

Limitations

Next experiment
```

This notebook becomes the bridge between the code and the research papers.

---

## 33. Example Questions You Should Eventually Be Able to Answer

### RoPE

Why does rotating Q/K encode positional relationships differently from adding learned position vectors?

### FlashAttention

What exactly is being avoided in memory, and why does that matter more at long sequence lengths?

### GELU vs SiLU

What change in the nonlinear function affects optimization?

### SwiGLU

Why does the gated FFN have more structure than simply swapping GELU for another activation?

### Adam vs AdamW

Why does decoupled weight decay matter?

### LoRA

Why can a low-rank update adapt a large pretrained model with far fewer trainable parameters?

The point of the laboratory is to make these questions answerable from both theory and experiment.

---

## 34. Definition of Done — Activation Study

- [ ] GELU baseline verified.
- [ ] ReLU verified.
- [ ] SiLU verified.
- [ ] SwiGLU verified.
- [ ] Parameter counts recorded.
- [ ] Parameter-matched comparison performed where appropriate.
- [ ] Learning curves generated.
- [ ] Throughput measured.
- [ ] Peak memory measured.
- [ ] Multiple seeds used for important conclusions.
- [ ] Results interpreted rather than merely ranked.

---

## 35. Definition of Done — Optimizer Study

- [ ] Adam understood.
- [ ] Adam reference implementation validated.
- [ ] AdamW understood.
- [ ] AdamW reference implementation validated.
- [ ] Weight-decay policy documented.
- [ ] Learning-rate policy documented.
- [ ] Optimizer-state memory measured.
- [ ] Learning curves compared.
- [ ] Tokens-to-target-loss compared.
- [ ] Wall-clock-to-target-loss compared.
- [ ] Final validation quality compared.
- [ ] Hyperparameter sensitivity documented.

---

## 36. Definition of Done — Attention/Position Study

- [ ] Vanilla causal attention verified.
- [ ] Learned absolute positions verified.
- [ ] RoPE verified.
- [ ] Fused/Flash attention path verified.
- [ ] All four combinations run from the same codebase.
- [ ] Same tokenizer and data split used.
- [ ] Same model dimensions used.
- [ ] Training tokens matched.
- [ ] Throughput compared.
- [ ] Memory compared.
- [ ] Validation quality compared.
- [ ] Context-length effects investigated.

---

## 37. Definition of Done — Full Laboratory

The framework is mature when:

- [ ] Attention is modular.
- [ ] Positional representation is modular.
- [ ] FFN/activation is modular.
- [ ] Optimizer is modular.
- [ ] Dataset is configurable.
- [ ] Tokenizer is configurable.
- [ ] Local development works.
- [ ] Colab works.
- [ ] Kaggle works.
- [ ] Checkpoint resume works.
- [ ] Every run saves configuration.
- [ ] Every run logs metrics.
- [ ] Benchmarks are automated.
- [ ] Results can be reproduced.
- [ ] Results can be plotted automatically.
- [ ] Experiments are organized by hypothesis.

---

## 38. Final Research Architecture

```text
                         TRANSFORMER LAB
                                │
          ┌─────────────────────┼─────────────────────┐
          │                     │                     │
      POSITION              ATTENTION                FFN
          │                     │                     │
     ┌────┴────┐          ┌─────┴─────┐        ┌─────┴─────┐
     │         │          │           │        │     │     │
 Learned     RoPE      Vanilla      Flash    GELU  SiLU  SwiGLU
 Absolute

                         TRAINING
                             │
                      ┌──────┴──────┐
                      │             │
                    Adam          AdamW
```

Then:

```text
Individual ablations
        ↓
Effects understood
        ↓
Interactions tested
        ↓
GPT-2-scale validation
        ↓
New research hypothesis
        ↓
Your own model
```

---

## 39. Final Principle

The objective is not to collect a table of whichever configuration gets the lowest validation loss.

The objective is to learn how to reason about a Transformer as a system.

For every component ask:

> What mathematical function does it implement?

> Why is it there?

> What assumption does it introduce?

> What happens if I remove it?

> What happens if I replace it?

> What does the replacement cost?

> How does it affect compute and memory?

> Does its effect survive when scale changes?

The eventual capability you are building is:

> **Read a research paper → understand its mathematical change → implement it from first principles → design a controlled experiment → measure the consequences → explain the result → formulate the next hypothesis.**
