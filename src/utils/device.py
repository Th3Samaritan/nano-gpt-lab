"""Device + dtype selection and GPU memory/throughput helpers.

WHAT
----
get_device() picks cuda -> mps -> cpu (plan section 23).
get_autocast_dtype() picks the mixed-precision dtype for the accelerator:
    bf16 on Ampere+ (bf16 capable), fp16 on older NVIDIA (T4, P100), and
    None (fp32) on CPU.
peak_memory_mb() reports peak allocated VRAM for efficiency logs.

WHY
----
Colab gives T4 (Turing, fp16-only for good mixed precision) while Kaggle may
give P100 or T4; A100/L4 support bf16. Choosing the wrong autocast dtype
silently changes numerics - and the plan explicitly forbids silently mixing
precision modes between experiments (section 16). The dtype actually used is
always recorded in environment.txt + config.yaml.

HOW
----
CUDA capability >= 8.0 means bf16 is a first-class citizen (Ampere).
Turing (7.5) and Volta/Pascal (7.0/6.0) get fp16 + GradScaler.
"""
from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn


def get_device() -> str:
    """Return 'cuda', 'mps' or 'cpu' depending on what this machine has."""
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def device_name() -> str:
    """Human-readable accelerator name for logs (e.g. 'Tesla T4')."""
    if torch.cuda.is_available():
        return torch.cuda.get_device_name(0)
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "Apple MPS"
    return "CPU"


def get_autocast_dtype(device: str) -> Optional[torch.dtype]:
    """Mixed-precision dtype for this accelerator, or None for fp32.

    bf16 keeps an fp32-sized exponent range so it needs no loss scaler;
    fp16 does (handled by training/precision.py).
    """
    if device == "cuda":
        if torch.cuda.is_bf16_supported():
            return torch.bfloat16
        return torch.float16
    return None


def peak_memory_mb() -> float:
    """Peak VRAM allocated by torch so far on this process (MB).

    Called at evaluation points so efficiency numbers are comparable across
    variants (plan section 17: peak GPU memory is a first-class metric).
    """
    if torch.cuda.is_available():
        return torch.cuda.max_memory_allocated() / (1024 * 1024)
    return 0.0


def reset_peak_memory() -> None:
    """Zero the peak-memory counter (call once before the training loop)."""
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()


def gpu_peak_flops(device: str) -> float:
    """Approximate peak fp16/bf16 FLOPS of common Colab/Kaggle GPUs.

    Used only for the MFU (model flops utilization) log line; it is a
    hardware table, not part of any experimental comparison.
    """
    if device != "cuda":
        return 0.0
    name = device_name().lower()
    table = {
        "a100": 312e12,
        "l4": 121e12,
        "v100": 125e12,
        "t4": 65e12,
        "p100": 21.2e12,
    }
    for key, flops in table.items():
        if key in name:
            return flops
    return 65e12  # unknown GPU: assume T4-class so MFU stays sane


# --------------------------------------------------------------------------
# Multi-GPU support (Kaggle offers 2x T4)
# --------------------------------------------------------------------------
class LogitsOnly(nn.Module):
    """Adapter: GPT.forward returns (logits, loss); nn.DataParallel gathers
    plain tensors (not tuples), so the parallel replica exposes a
    logits-only forward and the trainer computes the loss on the master.

    WHY this exists at all: the loss is a single scalar per device - if
    DataParallel had to gather losses, the trainer would receive a batch of
    scalars and the reduction semantics would be an extra thing to verify.
    Computing cross-entropy outside the model keeps ONE loss path for
    single-GPU and multi-GPU, i.e. zero behavioral difference."""

    def __init__(self, gpt: nn.Module):
        super().__init__()
        self.gpt = gpt

    def forward(self, idx):
        return self.gpt(idx)[0]


def resolve_device_ids(devices="auto") -> list:
    """Which CUDA devices to use, given config 'devices' (auto | int | list).

    'auto' uses every GPU the runtime offers (Kaggle T4x2 -> [0, 1]).
    An explicit int caps the count. Returns [] when no CUDA exists."""
    if not torch.cuda.is_available():
        return []
    n = torch.cuda.device_count()
    if n == 0:
        return []
    if devices == "auto" or devices is None:
        return list(range(n))
    if isinstance(devices, int):
        return list(range(min(devices, n)))
    if isinstance(devices, (list, tuple)):
        return [int(d) for d in devices if int(d) < n]
    return [0]


def wrap_parallel(model: nn.Module, device_ids: list):
    """Wrap in nn.DataParallel when device_ids has more than one entry.

    WHY DataParallel and not DDP here: a Kaggle/Colab notebook is a single
    process - DataParallel splits the micro-batch across the 2 GPUs and
    AVERAGES gradients before the optimizer step, which is the same update
    as one GPU with the full micro-batch (this model has no BatchNorm, so
    there are no cross-device statistics to worry about). The ablation
    stays intact: same effective batch, same recipe, ~2x throughput.
    Returns (model_or_wrapped, base_gpt)."""
    if len(device_ids) <= 1:
        return model, model
    wrapped = nn.DataParallel(LogitsOnly(model), device_ids=device_ids)
    return wrapped, model  # base_gpt stays the plain GPT for ckpt/generate


def get_base_gpt(model: nn.Module) -> nn.Module:
    """Recover the plain GPT from a possibly-wrapped model."""
    m = model.module if isinstance(model, nn.DataParallel) else model
    return m.gpt if isinstance(m, LogitsOnly) else m
