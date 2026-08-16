# NanoGPT → GPT-2 First-Principles Research Implementation Plan

## 1. Project Goal

Build a small, reproducible language-model research laboratory in PyTorch.

The project starts with a first-principles implementation of a NanoGPT-style decoder-only Transformer and progressively introduces:

1. Vanilla causal self-attention.
2. Learned absolute positional embeddings.
3. Rotary positional embeddings (RoPE).
4. FlashAttention / PyTorch fused scaled-dot-product attention.
5. BPE tokenization.
6. Controlled ablation experiments.
7. Scaling from a small model toward GPT-2.
8. Later investigation of GPT-3-style architectural ideas at manageable model sizes.

The central research question is not simply:

> Which implementation is faster?

It is:

> How do different attention and positional-information choices affect language-model quality, convergence, efficiency, memory usage, and scaling behavior when everything else is held constant?

---

# 2. Core Research Philosophy

The project should follow a strict first-principles workflow:

**Understand → implement → verify → benchmark → modify → compare → explain**

Do not begin with highly optimized implementations.

First understand the mathematical object being optimized. Then introduce the optimized implementation and measure what changes.

The guiding rule is:

> **One architectural variable at a time.**

Do not simultaneously change attention, tokenizer, optimizer, number of layers, context length, and dataset and then attribute the result to one modification.

---

# 3. Main Experimental Matrix

The first major experiment is a 2 × 2 design.

| Experiment | Attention | Positional method |
|---|---|---|
| A | Vanilla causal attention | Learned absolute position |
| B | Vanilla causal attention | RoPE |
| C | Flash/fused attention | Learned absolute position |
| D | Flash/fused attention | RoPE |

These four models should share the same:

- tokenizer
- vocabulary
- dataset split
- model depth
- embedding dimension
- number of heads
- context length
- initialization
- optimizer
- learning-rate schedule
- weight decay
- training-token budget
- evaluation procedure
- random seed policy

This creates a controlled ablation.

---

# 4. Datasets

Use the datasets progressively rather than training on all of them immediately.

## Dataset 1 — Tiny Shakespeare

Purpose:

- Verify the entire pipeline.
- Debug the Transformer.
- Understand training dynamics.
- Run rapid architecture experiments.
- Confirm that all four attention/position variants train.

This should be the first working dataset.

Do not optimize for benchmark quality here. Optimize for understanding and fast iteration.

---

## Dataset 2 — TinyStories

Purpose:

- Test whether observations from Shakespeare transfer to a more natural language distribution.
- Evaluate whether architecture changes affect generation quality more clearly.
- Stress the tokenizer and data pipeline.

Use a controlled subset initially so experiments can be repeated cheaply.

---

## Dataset 3 — OpenWebText

Purpose:

- Move toward a realistic GPT-style pretraining environment.
- Test whether observations from small datasets survive at larger scale.

Do not start with the entire dataset.

Begin with a fixed, documented subset.

Increase the data size only after the training pipeline, checkpointing, evaluation, and experiment tracking are stable.

---

# 5. Tokenization Strategy

Use BPE for the main experiments.

The tokenizer must remain identical across the four experimental variants.

Recommended progression:

### Phase 1
Use a known, stable BPE tokenizer to validate the model.

### Phase 2
Implement/inspect the tokenizer pipeline closely enough that you understand:

- vocabulary construction
- merges
- encoding
- decoding
- special tokens
- sequence packing
- end-of-document handling

Do not allow tokenizer differences to contaminate architecture comparisons.

Record:

- vocabulary size
- average tokens per document
- tokens/character
- special-token policy

---

# 6. Repository Structure

Use one codebase for both Colab and Kaggle.

```text
nano-gpt-lab/
│
├── README.md
├── IMPLEMENTATION_PLAN.md
├── requirements.txt
├── pyproject.toml
│
├── configs/
│   ├── shakespeare.yaml
│   ├── tinystories.yaml
│   ├── openwebtext.yaml
│   ├── exp_a_vanilla_learned.yaml
│   ├── exp_b_vanilla_rope.yaml
│   ├── exp_c_flash_learned.yaml
│   └── exp_d_flash_rope.yaml
│
├── data/
│   ├── download.py
│   ├── prepare_shakespeare.py
│   ├── prepare_tinystories.py
│   └── prepare_openwebtext.py
│
├── src/
│   ├── __init__.py
│   │
│   ├── model/
│   │   ├── embeddings.py
│   │   ├── rope.py
│   │   ├── attention.py
│   │   ├── flash_attention.py
│   │   ├── mlp.py
│   │   ├── normalization.py
│   │   ├── transformer_block.py
│   │   └── gpt.py
│   │
│   ├── tokenizer/
│   │   └── bpe.py
│   │
│   ├── training/
│   │   ├── trainer.py
│   │   ├── optimizer.py
│   │   ├── scheduler.py
│   │   └── precision.py
│   │
│   ├── evaluation/
│   │   ├── loss.py
│   │   ├── perplexity.py
│   │   └── generation.py
│   │
│   └── utils/
│       ├── checkpoint.py
│       ├── logging.py
│       ├── seed.py
│       ├── device.py
│       └── environment.py
│
├── scripts/
│   ├── train.py
│   ├── evaluate.py
│   ├── generate.py
│   └── benchmark.py
│
├── notebooks/
│   ├── 01_attention_from_scratch.ipynb
│   ├── 02_rope_from_scratch.ipynb
│   ├── 03_nanogpt_baseline.ipynb
│   └── 04_experiment_analysis.ipynb
│
├── checkpoints/
├── results/
├── logs/
└── runs/
```

The implementation should be executable from both notebooks and command line scripts.

---

# 7. Phase 0 — Environment and Reproducibility

Before implementing the model:

- Set up PyTorch.
- Detect CUDA automatically.
- Detect Colab vs Kaggle.
- Detect GPU name and VRAM.
- Record PyTorch version.
- Record CUDA version.
- Record Git commit/hash when available.
- Set random seeds.
- Make deterministic settings explicit.
- Create a run directory for every experiment.

Every run should produce:

```text
run/
├── config.yaml
├── metrics.csv
├── train.log
├── model_summary.txt
├── checkpoint.pt
└── environment.txt
```

A result without its configuration is not a reproducible result.

---

# 8. Phase 1 — Build Attention From First Principles

Implement causal self-attention without relying on FlashAttention.

Start from the mathematical operations.

Given hidden states:

```text
X
```

produce:

```text
Q = XWq
K = XWk
V = XWv
```

Then:

```text
S = QKᵀ / sqrt(d_head)
```

Apply a causal mask.

Then:

```text
A = softmax(S)
```

Finally:

```text
Y = AV
```

and apply the output projection.

Support:

- single-head attention first
- then multi-head attention
- then causal masking
- then dropout if required by the chosen baseline

Create unit tests for each stage.

---

# 9. Phase 2 — Implement Learned Positional Embeddings

Implement the standard learned absolute positional embedding baseline.

Conceptually:

```text
token_embedding + position_embedding
        ↓
Transformer
```

Verify:

- correct shape
- position range
- batch broadcasting
- parameter count
- causal behavior

This becomes **Experiment A**.

---

# 10. Phase 3 — Implement RoPE From First Principles

Implement Rotary Positional Embeddings yourself before using any optimized implementation.

RoPE should be applied to the query and key representations.

Conceptually:

```text
Q → rotate(Q, position)
K → rotate(K, position)
```

Do not add a learned position embedding at the input when using RoPE.

Create tests for:

- frequency construction
- sinusoidal terms
- dimension pairing
- rotation
- broadcasting across batch/head/sequence dimensions
- different sequence lengths

Verify that:

```text
RoPE attention ≠ learned absolute position attention
```

in implementation and parameter count where appropriate.

This becomes **Experiment B**.

---

# 11. Phase 4 — Implement Flash/Fused Attention

After the vanilla attention implementation is verified, add an optimized attention path.

Prefer PyTorch's supported fused/scaled-dot-product attention interface rather than starting with a custom CUDA kernel.

The important conceptual distinction is:

**Vanilla attention**

```text
QKᵀ
↓
materialized attention scores
↓
softmax
↓
multiply V
```

**Flash/fused attention**

The implementation avoids the same expensive materialization pattern and uses a memory-efficient kernel path where available.

Benchmark:

- tokens/sec
- peak GPU memory
- training step time
- forward time
- backward time

Do not expect FlashAttention to inherently improve validation loss. Its primary purpose is computational and memory efficiency.

This becomes **Experiment C** when paired with learned positions.

---

# 12. Phase 5 — Combine Flash Attention + RoPE

Combine:

```text
RoPE
+
Flash/fused attention
```

Keep everything else fixed.

This becomes **Experiment D**.

Now the complete 2 × 2 matrix exists.

---

# 13. NanoGPT Model Configuration

Start with a small model instead of immediately building 124M parameters.

Example research model:

```yaml
block_size: 256
n_layer: 4
n_head: 4
n_embd: 256
dropout: 0.0
bias: true
```

The exact values are not sacred.

The purpose of the first model is rapid experimentation.

Once the implementation is verified, increase the size.

---

# 14. Training Configuration

Use a common training configuration for all four experiments.

Record at minimum:

```text
micro_batch_size
gradient_accumulation_steps
effective_batch_size
block_size
learning_rate
minimum_learning_rate
warmup_steps
decay_steps
weight_decay
gradient_clipping
optimizer
dtype
total_training_tokens
evaluation_interval
checkpoint_interval
seed
```

A useful starting pattern on a T4 is:

```text
micro-batch:
4 → 2 → 1 if necessary

context:
1024 → 512 → 256 if necessary

gradient accumulation:
increase as micro-batch decreases
```

The actual values should be determined by a memory benchmark rather than assumed.

---

# 15. Effective Batch Size

Distinguish clearly between:

**Micro-batch size**

What physically fits on the GPU.

**Gradient accumulation steps**

How many micro-batches are processed before updating.

**Effective batch size**

Approximately:

```text
micro_batch
×
gradient_accumulation
```

and in token terms:

```text
micro_batch
×
gradient_accumulation
×
sequence_length
```

Log both examples.

Do not report only batch size without clarifying which definition is being used.

---

# 16. Mixed Precision

For NVIDIA T4 experiments:

- use mixed precision where appropriate
- use FP16/autocast with the appropriate scaler or PyTorch training mechanism
- verify numerical stability
- keep loss computation stable
- record the precision mode in the run configuration

Do not silently mix precision modes between experiments.

---

# 17. First Benchmark — Shakespeare

Run all four variants.

Use exactly the same:

- training tokens
- validation set
- random seed policy
- tokenizer
- context length
- batch policy
- optimizer settings

Example:

```text
A: Vanilla + Learned Position
B: Vanilla + RoPE
C: Flash + Learned Position
D: Flash + RoPE
```

Collect:

```text
final train loss
best validation loss
validation perplexity
tokens/sec
average step time
peak GPU memory
total training time
parameter count
checkpoint size
```

Also save generated samples using identical prompts and generation parameters.

---

# 18. Do Not Compare Only Final Loss

Plot:

### Learning curves

```text
training loss vs tokens
validation loss vs tokens
```

### Efficiency

```text
tokens/sec
peak VRAM
step time
```

### Convergence

Record how many tokens each experiment needs to reach a fixed validation-loss threshold.

This allows questions such as:

> Did RoPE reach the same quality with fewer training tokens?

and:

> Did FlashAttention reduce training time without changing learning behavior?

---

# 19. Statistical/Repetition Policy

A single random seed is useful for debugging but weak evidence for a research claim.

For final conclusions:

- use multiple seeds for important comparisons
- report mean and standard deviation when practical
- keep a fixed seed set across variants

Example:

```text
seed = 42
seed = 123
seed = 2026
```

Do not average away anomalous runs silently.

Keep failed runs recorded.

---

# 20. Phase 6 — TinyStories

Repeat the exact experiment matrix.

Do not modify the architecture simply because a previous result was surprising.

First reproduce the same A/B/C/D comparison.

Then ask whether the observations from Shakespeare persist.

Questions:

- Does RoPE consistently improve validation loss?
- Does FlashAttention change throughput similarly?
- Does the ranking of variants change with dataset complexity?
- Does context length affect the relative performance?

---

# 21. Phase 7 — OpenWebText Subset

Start with a fixed subset.

For example:

```text
subset_1
subset_2
subset_3
```

but only increase the dataset size after the pipeline is stable.

Measure:

- tokens seen
- throughput
- memory
- validation loss
- perplexity
- scaling behavior

The goal is to discover whether the small-data conclusions survive in a more realistic pretraining setting.

---

# 22. Experiment Tracking

Every run should have a unique ID.

Example:

```text
shakespeare/
├── A_vanilla_learned_seed42/
├── B_vanilla_rope_seed42/
├── C_flash_learned_seed42/
└── D_flash_rope_seed42/
```

Use a consistent naming convention:

```text
{dataset}_{attention}_{position}_{modelsize}_{seed}
```

Optional experiment trackers can be added later, but plain CSV/JSON logging should always remain available.

---

# 23. Hardware Abstraction

The code should automatically select:

```text
CUDA → GPU
MPS → Apple Silicon
CPU → fallback
```

For the primary experiments, optimize for NVIDIA CUDA because the target environments are Colab and Kaggle.

At startup print:

```text
Environment
GPU
VRAM
PyTorch version
CUDA version
dtype
batch size
context length
parameter count
```

---

# 24. Colab Support

The same repository should run in Colab with minimal notebook-specific code.

On Colab:

1. Mount Google Drive.
2. Clone/download the repository.
3. Detect the mounted Drive path.
4. Save:
   - checkpoints
   - logs
   - metrics
   - generated samples
   - experiment configs
   - plots
5. Resume interrupted training from the latest checkpoint.

Example directory:

```text
/content/drive/MyDrive/nano-gpt-lab/
```

Do not make Drive the primary hot training directory if I/O becomes a bottleneck. Train using local runtime storage and periodically copy checkpoints/results to Drive.

---

# 25. Kaggle Support

The same repository should run in Kaggle.

Use:

```text
/kaggle/working/
```

for active experiment outputs.

Keep source data in Kaggle's supported input/data mechanisms where practical.

Save final artifacts under the working directory.

The project should detect Kaggle automatically and never require changing model code.

---

# 26. Checkpointing

Checkpoint at regular intervals.

A checkpoint should contain:

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

This allows interrupted Colab/Kaggle sessions to resume accurately.

Also keep:

```text
best.pt
latest.pt
```

---

# 27. Evaluation Pipeline

Create a single evaluation function used by every experiment.

Report:

```text
loss
perplexity
tokens/sec
peak VRAM
```

Generation should use identical:

```text
temperature
top-k
top-p
max_new_tokens
prompt
```

when comparing samples.

Never manually select the best-looking generated text as the primary evaluation.

---

# 28. Performance Benchmarking

Separate three concepts:

### Model quality

- validation loss
- perplexity
- generation samples

### Training efficiency

- tokens/sec
- steps/sec
- wall-clock time
- peak VRAM

### Inference efficiency

- tokens/sec
- latency
- memory

This matters because a method can be:

- more accurate but slower
- equally accurate but faster
- faster and more memory efficient
- slightly worse in loss but much cheaper

There is no single "best" metric.

---

# 29. Unit Tests

Before serious training, test:

### Attention

- shape correctness
- causal masking
- no future-token leakage

### RoPE

- correct shape
- deterministic transformation
- correct rotation structure
- works for multiple sequence lengths

### Transformer

- residual connections
- normalization
- output shape

### Tokenizer

- encode/decode consistency
- special tokens

### Training

- loss decreases on a tiny batch
- model can overfit a tiny dataset

A very important test:

> Train on a tiny dataset until near-zero training loss.

If the model cannot overfit a tiny dataset, do not proceed to large-scale training.

---

# 30. First-Principles "Naked" Implementations

The first implementation of each major component should avoid hiding the core mathematics behind a high-level library.

Implement explicitly:

```text
Q
K
V
QKᵀ
scale
causal mask
softmax
AV
projection
```

Then compare your implementation against the optimized PyTorch implementation.

The purpose is not to permanently use the slow implementation.

The purpose is to understand what the optimized implementation is replacing.

---

# 31. Validation Against PyTorch

Once vanilla attention is implemented:

1. Generate the same Q/K/V tensors.
2. Run your attention.
3. Run the PyTorch reference implementation.
4. Compare outputs.
5. Compare gradients.
6. Measure numerical error.

Do the same for RoPE.

Only after the numerical behavior is validated should you run large training experiments.

---

# 32. Research Notebook Strategy

Maintain a research notebook documenting:

```text
Hypothesis
Implementation
Expected behavior
Observed behavior
Explanation
Next experiment
```

Example:

```text
Hypothesis:
RoPE will improve positional generalization relative to learned absolute embeddings.

Test:
Same NanoGPT architecture, same data, same training budget.

Result:
Validation loss improved by X%.

Observation:
Effect became larger/smaller at longer context.

Interpretation:
...

Next test:
Increase context length while holding parameter count fixed.
```

This turns the project from a collection of coding experiments into a research program.

---

# 33. Phase 8 — Scale Toward GPT-2

After the NanoGPT experiments are reliable:

Move toward the GPT-2 small configuration:

```text
~124M parameters
12 layers
12 heads
hidden size ~768
context length ~1024
```

Do not immediately attempt to reproduce the original full training compute.

First validate:

```text
architecture
parameter count
forward pass
backward pass
memory requirements
checkpointing
tokenizer
training loop
evaluation
```

Then establish a smaller training budget suitable for Colab/Kaggle.

The research objective is reproducible experimentation, not pretending that a single T4 reproduces the original GPT-2 training run.

---

# 34. GPT-2 Ablation Matrix

Once the 124M-class implementation is stable, repeat selected experiments:

```text
GPT-2 baseline
GPT-2 + RoPE
GPT-2 + Flash/Fused Attention
GPT-2 + RoPE + Flash/Fused Attention
```

At this stage, track scaling effects carefully.

Questions become:

- Does an improvement observed at NanoGPT scale persist at 124M?
- Does RoPE interact differently with larger models?
- Does FlashAttention become more valuable as context length increases?
- Does memory savings enable a larger batch or context length?
- Does the quality/efficiency trade-off change with model size?

---

# 35. Phase 9 — Beyond GPT-2

Only after the 124M-scale experiments are understood should you investigate GPT-3-style ideas.

Do not interpret "GPT-3" as "train GPT-3."

Instead:

> Reproduce the architectural/training principles at a scale you can afford.

Potential future experiments:

```text
LayerNorm → RMSNorm
GELU → SwiGLU
MHA → GQA
MHA → MQA
RoPE variations
longer context
different initialization
different optimizers
different learning-rate schedules
different scaling laws
```

Again:

**one change at a time.**

---

# 36. Experimental Priority Order

The recommended order is:

```text
1. PyTorch environment
2. BPE tokenizer
3. Dataset pipeline
4. Vanilla attention
5. Learned positional embeddings
6. Small NanoGPT
7. Tiny-dataset overfit test
8. Shakespeare baseline
9. RoPE from scratch
10. Shakespeare + RoPE
11. PyTorch fused/Flash attention
12. Shakespeare + Flash
13. Shakespeare + Flash + RoPE
14. Controlled A/B/C/D analysis
15. TinyStories
16. OpenWebText subset
17. NanoGPT scaling
18. GPT-2 architecture
19. GPT-2 ablations
20. GPT-3-style architectural experiments
```

Do not skip the validation steps simply because the model appears to train.

---

# 37. Definition of Done for NanoGPT

Do not move to GPT-2 until all of the following are true:

- [ ] BPE tokenizer works.
- [ ] Vanilla attention is implemented from first principles.
- [ ] Causal masking is verified.
- [ ] Learned position embeddings work.
- [ ] RoPE is implemented and tested.
- [ ] Fused/Flash attention path works.
- [ ] Model can overfit a tiny dataset.
- [ ] Shakespeare training completes.
- [ ] All four experiments run from one codebase.
- [ ] Checkpoints resume successfully.
- [ ] Colab saves results to Drive.
- [ ] Kaggle runs without model-code modifications.
- [ ] Metrics are automatically logged.
- [ ] Results can be reproduced from the saved configuration.
- [ ] Training/inference throughput is benchmarked.
- [ ] At least one controlled comparison has been completed.

---

# 38. Definition of Done for the First Research Study

The first study is complete when you can produce one reproducible report containing:

## Experimental setup

- model configuration
- tokenizer
- datasets
- training budget
- hardware
- precision
- random seeds

## Results

- validation loss
- perplexity
- training time
- tokens/sec
- peak memory
- parameter count

## Comparison

```text
Vanilla + Learned
Vanilla + RoPE
Flash + Learned
Flash + RoPE
```

## Analysis

Explain:

- what changed
- what did not change
- why you expected the result
- what actually happened
- possible causes
- limitations
- what experiment should come next

---

# 39. The Main Rule

Do not optimize the project around getting a good-looking number.

Optimize it around being able to answer:

> **Why did the number change?**

That is the central principle of this project.

You are not merely implementing NanoGPT.

You are building a controlled environment in which you can decompose a Transformer into understandable components, reconstruct those components yourself, replace one assumption at a time, and empirically investigate the consequences.

---

# 40. Final Roadmap

```text
FIRST PRINCIPLES
        ↓
Attention from scratch
        ↓
Learned positional embedding
        ↓
RoPE from scratch
        ↓
Fused/Flash attention
        ↓
NanoGPT
        ↓
BPE
        ↓
Shakespeare
        ↓
        ┌──────────────────────────────┐
        │  A Vanilla + Learned         │
        │  B Vanilla + RoPE            │
        │  C Flash + Learned           │
        │  D Flash + RoPE              │
        └──────────────────────────────┘
        ↓
Controlled ablation
        ↓
TinyStories
        ↓
OpenWebText subset
        ↓
Scale model
        ↓
GPT-2-class ~124M
        ↓
Repeat controlled ablations
        ↓
Study GPT-3-style architectural ideas
        ↓
Form your own hypothesis
        ↓
Build your own model
```

## End Goal

The end goal is not merely:

> "I implemented GPT-2."

The end goal is:

> **"I can take a language-model architecture apart, derive and implement its components, understand the trade-offs, modify one of its assumptions, run a controlled experiment, analyze the result, and use that evidence to design the next experiment."**

That is the capability this project is designed to build.
