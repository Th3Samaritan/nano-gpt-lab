"""nano-gpt-lab: first-principles NanoGPT -> GPT-2 research laboratory.

Package layout mirrors the implementation plan (see IMPLEMENTATION_PLAN.md):
    src/model        - embeddings, RoPE, vanilla attention, flash attention,
                       MLP, normalization, transformer block, GPT assembly
    src/tokenizer    - byte-pair encoding trained from scratch
    src/training     - optimizer, LR scheduler, mixed precision, training loop
    src/evaluation   - unified loss / perplexity / generation functions
    src/utils        - environment detection, seeds, device, logging,
                       checkpoints, config loading

Nothing here depends on Colab or Kaggle specific packages; the platform
differences are isolated in src/utils/environment.py.
"""
