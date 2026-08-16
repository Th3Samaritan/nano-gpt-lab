# nano-gpt-lab

A first-principles NanoGPT → GPT-2 research laboratory. One codebase, two
tiers, one scientific question.

```
THE QUESTION
"Which recipe learns language best, and what does the win cost?"
    A  Classic attention + learned position table   (GPT-2 baseline)
    B  Classic attention + RoPE rotation
    C  Flash attention   + learned position table
    D  Flash attention   + RoPE rotation
Everything else - tokenizer, data, depth, optimizer, schedule, seeds -
is identical. One variable at a time (IMPLEMENTATION_PLAN.md, section 39:
the goal is answering WHY the number changed).
```

---

## The two-tier workflow

| Tier | Where | What happens there |
|---|---|---|
| **Local machine (your PC)** | this repo | First-principles implementation, unit tests, BPE/RoPE/attention from scratch, tiny models, debugging, experiment development |
| **Colab/Kaggle T4** | `notebooks/colab_kaggle_lab.ipynb` | Larger NanoGPT, TinyStories, OpenWebText subset, FlashAttention benchmarks, GPT-2 ~124M runs |

Nothing in `src/` knows which tier it is on - the only platform-aware file
is `src/utils/environment.py`.

### Kaggle T4x2 (both GPUs, ~2x speed)

Kaggle also offers a 2x T4 accelerator. When it is present, training
automatically wraps the model in `nn.DataParallel`: the micro-batch is
split across the two GPUs and the gradients are averaged before the
optimizer step - the same update as one GPU with the full micro-batch
(no BatchNorm here, so no cross-device statistics). Same recipe, same
effective batch, roughly half the wall-clock. Control it with
`--devices auto | 1 | 2` (or `DEVICES` in the notebook); the devices
actually used are recorded in every run's `config.yaml` and
`environment.txt`. Checkpoints always store the plain model (no
`module.` prefixes), so a T4x2 run resumes fine on a single T4 and
vice versa.

---

## Quick start (local machine)

```bash
# 1. deps (torch + numpy + pyyaml + tqdm + matplotlib + requests)
pip install -r requirements.txt

# 2. the test suite - including THE GATE (tiny-corpus overfit, ~1 min CPU)
python scripts/run_tests.py            # or: python -m pytest tests -q

# 3. first training: experiment A, tiny model, 400 steps
python scripts/train.py --dataset shakespeare --exp exp_a_vanilla_learned --steps 400 --seeds 42

# 4. the full A/B/C/D comparison (nano size - CPU-friendly)
python scripts/train.py --dataset shakespeare --compare --steps 2000 --seeds 42

# 5. the layman reports (PNG charts + comparison table)
python scripts/analyze.py --dataset shakespeare --size nano
```

### Colab

1. Open `notebooks/colab_kaggle_lab.ipynb` in Colab
   (File → Open notebook → GitHub tab → `Th3Samaritan/nano-gpt-lab`).
2. Run the cells. The notebook fetches the repo itself via `git clone`
   (zipball fallback) - no zip upload needed.
3. Cell 4: set `SIZE = "gpt2"`, `STEPS = 2000`; run the comparison.
4. Checkpoints + logs + charts are mirrored to
   `/content/drive/MyDrive/nano-gpt-lab/` every checkpoint interval (Drive
   is mounted automatically; training itself runs on fast runtime storage,
   plan section 24).

### Kaggle

1. Create a notebook, attach GPU T4, and paste
   `notebooks/colab_kaggle_lab.ipynb`'s content (or upload the .ipynb).
2. Run: the notebook `git clone`s the repo into `/kaggle/working`
   (Kaggle notebooks have internet; no Dataset input needed).
3. Everything lands in `/kaggle/working/nano-gpt-lab/` - persisted to your
   Kaggle account automatically as the notebook output.

### GitHub

Repository: https://github.com/Th3Samaritan/nano-gpt-lab

```bash
# after changing code on the local machine:
cd nano-gpt-lab
git add -A && git commit -m "describe the change" && git push

# on Colab/Kaggle, pull the update before a new session:
git pull origin main
```

---

## What is in here (the what/why/where of every piece)

### First-principles components - built naked, then verified

| File | What it implements | The key idea |
|---|---|---|
| `src/tokenizer/bpe.py` | Byte-level BPE trained from scratch | Repeatedly merge the most frequent byte pair; numpy `bincount` makes the textbook algorithm run on 10MB corpora without approximation |
| `src/model/attention.py` | Vanilla causal attention | The naked math: `softmax(QK^T/sqrt(d))·V` with an explicit T×T score table |
| `src/model/flash_attention.py` | FlashAttention, both ways | `flash_tiled`: the paper's Algorithm 1 (online softmax) in ~25 lines of PyTorch - tile loop, running max, rescale; `flash_fused`: torch's CUDA kernels via `scaled_dot_product_attention` |
| `src/model/rope.py` | Rotary position embeddings | Position = rotating pairs of q/k dims; `⟨R(m)q, R(n)k⟩ = ⟨R(m−n)q, k⟩` so attention sees only relative position |
| `src/model/embeddings.py` | Learned position table | The GPT-2 baseline that RoPE replaces |
| `src/model/gpt.py` | GPT-2 assembly | Pre-LN blocks, weight tying, GPT-2 init incl. residual scaling |
| `src/training/*` | Optimizer / schedule / precision / loop | Decay/no-decay AdamW split, warmup+cosine, fp16-scaler on T4 & bf16 on Ampere+, grad accumulation, resume-exact checkpoints |
| `src/evaluation/*` | One eval function for everything | Same estimator, same sampler settings for A/B/C/D (plan section 27) |

### Validation before every experiment

- `tests/test_attention.py` - shapes, **no future leakage**, and the wall:
  `tiled == vanilla` and `fused == vanilla` to tight tolerance.
- `tests/test_rope.py` - the relative-position identity, norm preservation.
- `tests/test_tokenizer.py` - roundtrip, compression, determinism, unicode.
- `tests/test_transformer.py` - all five variant combos, weight tying,
  exact parameter counts, gradient flow.
- `tests/test_overfit.py` - **THE GATE**: tiny corpus → loss ≈ 0 in a
  minute. If this fails, do not train big models (plan section 29).

### The four experiments

| File | Experiment | Attention | Position |
|---|---|---|---|
| `configs/exp_a_vanilla_learned.yaml` | A (control) | `vanilla` | `learned` |
| `configs/exp_b_vanilla_rope.yaml` | B | `vanilla` | `rope` |
| `configs/exp_c_flash_learned.yaml` | C | `flash_fused` | `learned` |
| `configs/exp_d_flash_rope.yaml` | D | `flash_fused` | `rope` |

Swap `flash_fused` → `flash_tiled` in C/D to train with the pure-PyTorch
first-principles tiler (slower, educational, numerically identical).
The training recipe in all four files is byte-identical - a run's own
`config.yaml` is saved inside its run dir, so every result is reproducible
(plan section 7: a result without its configuration is not a result).

### Datasets

| Config | Dataset | Tier | Notes |
|---|---|---|---|
| `configs/shakespeare.yaml` | Tiny Shakespeare (1.1MB) | local machine | Pipeline debugger + rapid A/B/C/D |
| `configs/tinystories.yaml` | TinyStories **50MB documented subset** | T4 | Natural-language transfer test |
| `configs/openwebtext.yaml` | OpenWebText **100MB documented subset** | T4 | Realistic pretraining (needs `pip install datasets` or a local file) |

BPE is trained **from scratch per dataset** (vocab 4096 Shakespeare /
10000 the rest), cached to `data/<dataset>/*.bin`, and one tokenizer is
shared by all four variants of that dataset.

### Model size presets (`src/utils/config.py`)

`nano` (4L/4H/256, block 256) → `small` (6L/6H/384) → `gpt2` (12L/12H/768,
block 1024, ~124M) → `gpt3` (GPT-3 125M recipe: block 2048, 0.5M-token
batches). `--size` changes architecture only - never the experimental
variables.

---

## The layman report suite

`scripts/analyze.py` (or the last cell of the runner notebook) produces
four charts in `results/<dataset>/` designed to be readable in 5 seconds:

| Chart | Shows | How to read it |
|---|---|---|
| `01_which_recipe_learned_best.png` | Best val loss per variant | Lower bar = better learner; the star is the winner |
| `02_speed_and_memory.png` | tok/s + peak GPU MB | Higher = faster; lower = less memory |
| `03_learning_speed.png` | Tokens needed to hit a fixed loss | Shorter bar = learned faster (the "RoPE needs fewer tokens?" question) |
| `04_verdict_card.png` | The 2×2 poster with all numbers | One glance: who won, by how much, at what cost |

Concept cartoons (also in the notebooks): `plot_attention_concept`,
`plot_rope_concept`, `plot_flash_concept`, `plot_bpe_concept` in
`src/utils/plots.py`.

Every run dir additionally contains: `config.yaml`, `environment.txt`,
`model_summary.txt`, `metrics.csv`, `train.log`, `samples.txt`,
`loss_curves.png`, `latest.pt`, `best.pt`, `summary.json`,
`tokenizer.json`.

---

## Command reference

```bash
# train one experiment            (one seed)
python scripts/train.py --dataset shakespeare --exp exp_b_vanilla_rope --steps 2000 --seeds 42

# the controlled comparison       (all four, or multiple seeds)
python scripts/train.py --dataset shakespeare --compare --seeds 42 123 2026

# scaled tier                     (T4: 124M model)
python scripts/train.py --dataset tinystories --compare --size gpt2 --steps 2000

# resume after a disconnect       (exact: optimizer, schedule, scaler, RNG)
python scripts/train.py --dataset shakespeare --exp exp_a_vanilla_learned --resume

# reports
python scripts/analyze.py --dataset shakespeare --size nano --threshold 2.0

# evaluate / sample a finished run
python scripts/evaluate.py --run-dir runs/shakespeare/flash_fused_rope_nano_seed42
python scripts/generate.py --run-dir runs/shakespeare/flash_fused_rope_nano_seed42 --prompt "To be, or not to be"

# feasibility scan before long runs (plan section 14)
python scripts/benchmark.py --dataset shakespeare --exp exp_d_flash_rope --size gpt2

# tests (both runners supported)
python scripts/run_tests.py
python -m pytest tests -q
```

Notebooks: `01_attention_from_scratch`, `02_rope_from_scratch`,
`03_nanogpt_baseline` (gate + first run), `04_experiment_analysis`,
`colab_kaggle_lab` (the universal runner).

---

## Interpreting results - the questions to ask

1. **A vs C** and **B vs D** (vanilla → flash): loss should NOT change.
   FlashAttention is an exact algorithm - it buys speed and memory, not
   quality. If loss changed, something leaked.
2. **A vs B** and **C vs D** (learned → RoPE): does quality change at
   equal tokens? Does `03_learning_speed.png` show fewer tokens to the
   same loss? Does the effect grow with context length?
3. **Does any ranking flip on TinyStories?** (plan section 20)
4. Every observation goes into the report template: hypothesis → setup →
   result → why you expected it → what happened → limitation → next
   experiment (plan section 32).

### First local machine result (interim - 250 steps, nano, seed 42)

| Variant | Best val loss | PPL |
|---|---|---|
| A vanilla+learned | 6.306 | 548 |
| B vanilla+RoPE | **6.185** | **485** |
| C flash+learned | 6.306 | 548 |
| D flash+RoPE | **6.185** | **485** |

Two clean findings, exactly what first principles predict:
1. Flash == vanilla bit-for-bit (6.306 vs 6.306) - the fused kernels
   change nothing about quality, only speed/memory.
2. RoPE beats the learned position table by 0.12 loss (~2%) at equal
   tokens, with 65k fewer parameters.

Caveats: one seed, CPU fp32, 2M tokens, tiny model. The T4 `gpt2` runs
(2000+ steps, seeds 42/123/2026) are the real experiment.

## Roadmap (from the plan, section 36)

local machine: env → BPE → data → vanilla attention → learned positions → tiny
NanoGPT → **overfit gate** → Shakespeare baseline → RoPE → fused flash →
Shakespeare A/B/C/D → analysis.
T4: TinyStories matrix → OpenWebText subset → scale to GPT-2 124M →
repeat ablations → GPT-3-style ideas (RMSNorm, SwiGLU, GQA, longer
context), still one change at a time.

## Troubleshooting

- **`OMP: Error #15` on Windows/conda**: duplicate OpenMP runtimes (torch's
  vs MKL's). All entry points set `KMP_DUPLICATE_LIB_OK=TRUE`; if you write
  your own script, set it before importing torch/numpy.
- **Shakespeare 404**: the old nanoGPT data URL is dead upstream; the lab
  uses the canonical karpathy/char-rnn mirror.
- **First Colab run slow**: the BPE + tokenize cache is stored in runtime
  storage; the second session re-tokenizes (~2-5 min). The trained BPE
  (`tokenizer.json`) is mirrored to Drive inside each run dir.
- **OOM on T4**: run `scripts/benchmark.py --size gpt2` and take its
  recommendation (micro-batch 4→2→1, block 1024→512, plan section 14).
