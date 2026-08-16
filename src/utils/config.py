"""Config loading/merging and model-size presets.

WHAT
----
The lab is configured by YAML files (configs/*.yaml) with three layers:
    dataset yaml   - data URLs, vocab size, BPE sample, eval prompts
    experiment yaml- ONE architectural variable: attention x position,
                     plus the shared training recipe
    CLI overrides  - quick knobs like --steps 500 (never architectural)

build_run_config() merges dataset + experiment + overrides into the single
self-contained dict that is saved as config.yaml inside every run dir. A
result without its configuration is not a reproducible result (plan sec 7).

WHY
----
Splitting data from experiment from CLI keeps the ablation matrix honest:
changing an experiment means editing ONE yaml, and every saved config.yaml
lets a future run (or reviewer) rebuild the exact same model.

HOW
----
deep_merge() recursively overlays dicts (later wins). MODEL_SIZES holds the
scale ladder from the plan: nano (first experiments) -> small -> gpt2
(124M-class, section 33) -> gpt3 (125M architecture with GPT-3 recipe,
section 35). --size patches only the architectural subset of keys, so the
experiment's attention/position choice is never touched by scaling.
"""
from __future__ import annotations

import copy
import os
from typing import Any, Optional

import yaml


def load_yaml(path: str) -> dict:
    """Load a YAML file into a dict (None -> empty dict)."""
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return data if isinstance(data, dict) else {}


def save_yaml(data: dict, path: str) -> None:
    """Write a dict as YAML (used for the config.yaml inside run dirs)."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, sort_keys=False, default_flow_style=False)


def deep_merge(base: dict, override: dict) -> dict:
    """Recursively overlay override onto base; override wins per key."""
    out = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = deep_merge(out[key], value)
        else:
            out[key] = copy.deepcopy(value)
    return out


# --------------------------------------------------------------------------
# Model size presets (plan sections 13 / 33 / 35)
# --------------------------------------------------------------------------
# "nano"  : fast-iteration research model - block 256, 4L/4H/256.
# "small" : mid-point used to check observations survive one size jump.
# "gpt2"  : GPT-2 small class - 12L/12H/768, context 1024 (~124M params).
# "gpt3"  : GPT-3 125M recipe - same architecture as gpt2 but context 2048
#           and the GPT-3 optimizer recipe (lr 6e-4, cosine to 6e-5,
#           0.5M-token batches) - the Phase-9 stepping stone.
MODEL_SIZES = {
    "nano": {
        "block_size": 256,
        "n_layer": 4,
        "n_head": 4,
        "n_embd": 256,
        "micro_batch_size": 8,
        "gradient_accumulation_steps": 4,
        "learning_rate": 3e-4,
        "min_learning_rate": 3e-5,
        "warmup_steps": 100,
        "max_steps": 2000,
    },
    "small": {
        "block_size": 512,
        "n_layer": 6,
        "n_head": 6,
        "n_embd": 384,
        "micro_batch_size": 8,
        "gradient_accumulation_steps": 4,
        "learning_rate": 3e-4,
        "min_learning_rate": 3e-5,
        "warmup_steps": 100,
        "max_steps": 3000,
    },
    "gpt2": {
        "block_size": 1024,
        "n_layer": 12,
        "n_head": 12,
        "n_embd": 768,
        "micro_batch_size": 4,
        "gradient_accumulation_steps": 8,
        "learning_rate": 6e-4,
        "min_learning_rate": 6e-5,
        "warmup_steps": 100,
        "max_steps": 3000,
    },
    "gpt3": {
        "block_size": 2048,
        "n_layer": 12,
        "n_head": 12,
        "n_embd": 768,
        "micro_batch_size": 4,
        "gradient_accumulation_steps": 16,
        "learning_rate": 6e-4,
        "min_learning_rate": 6e-5,
        "warmup_steps": 100,
        "max_steps": 3000,
    },
}


def configs_dir() -> str:
    """Absolute path to the configs/ directory."""
    from src.utils.environment import repo_root
    return os.path.join(repo_root(), "configs")


def load_dataset_config(name: str) -> dict:
    """Load configs/<name>.yaml (the dataset layer)."""
    return load_yaml(os.path.join(configs_dir(), f"{name}.yaml"))


def load_experiment_config(name: str) -> dict:
    """Load configs/<name>.yaml (the experiment layer: exp_a .. exp_d)."""
    return load_yaml(os.path.join(configs_dir(), f"{name}.yaml"))


def build_run_config(
    dataset_name: str,
    experiment_name: str,
    size: str = "nano",
    overrides: Optional[dict] = None,
) -> dict:
    """Merge dataset + experiment + size preset + CLI overrides.

    Precedence (highest last): dataset < experiment < size preset < overrides.
    The size preset intentionally overwrites experiment values for the
    architectural knobs ONLY - attention/position (the experiment variables)
    live only in the experiment yaml and are never touched here.
    """
    dataset_cfg = load_dataset_config(dataset_name)
    exp_cfg = load_experiment_config(experiment_name)
    if size not in MODEL_SIZES:
        raise KeyError(f"unknown model size '{size}', pick one of {list(MODEL_SIZES)}")
    merged = deep_merge({}, dataset_cfg)
    merged = deep_merge(merged, exp_cfg)
    merged = deep_merge(merged, {"size": size})
    merged = deep_merge(merged, MODEL_SIZES[size])
    if overrides:
        merged = deep_merge(merged, overrides)
    merged["dataset"] = dataset_name
    merged["experiment"] = experiment_name
    return merged
