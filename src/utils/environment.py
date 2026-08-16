"""Platform / environment abstraction.

WHAT
----
Detects whether the code is running on Google Colab, Kaggle, or a local
machine, and resolves three path concepts:

    runtime root  - where training actually happens (fast local storage)
    mirror root   - where artifacts are persisted (Drive on Colab; the same
                    runtime dir on Kaggle because /kaggle/working itself is
                    snapshotted to the user's Kaggle account at session end)
    data dir      - raw text + tokenized .bin caches (big, regenerable files)

WHY
---
Colab storage mounted from Drive is too slow for hot training I/O but is the
only thing that survives a disconnected session. Kaggle persists everything
under /kaggle/working automatically. Local runs just use the repo directory.
Centralizing this here means the model/training code never contains a single
platform branch - the plan requires Kaggle to run without model-code changes.

HOW
---
get_platform() inspects sys.modules for google.colab and the
KAGGLE_KERNEL_RUN_TYPE env var. The repo is expected at a fixed location on
each platform so path resolution needs no configuration:
    Colab : /content/nano-gpt-lab
    Kaggle: /kaggle/working/nano-gpt-lab
    Local : <repo root, i.e. parent of src/>
"""
from __future__ import annotations

import os
import platform as _platform
import shutil
import socket
import subprocess
import sys
import time

try:
    import torch
except ImportError:  # pragma: no cover - torch is required for the lab
    torch = None


# --------------------------------------------------------------------------
# Platform detection
# --------------------------------------------------------------------------
def get_platform() -> str:
    """Return 'colab', 'kaggle' or 'local'.

    Kaggle sets KAGGLE_KERNEL_RUN_TYPE only inside an interactive kernel.
    Colab injects the google.colab module into sys.modules at boot.
    """
    try:
        import google.colab  # noqa: F401  (import only to test presence)
        return "colab"
    except ImportError:
        pass
    if os.environ.get("KAGGLE_KERNEL_RUN_TYPE"):
        return "kaggle"
    return "local"


PLATFORM = get_platform()

# Fixed repo locations per platform (see module docstring for the why).
REPO_NAME = "nano-gpt-lab"
_COLAB_REPO = f"/content/{REPO_NAME}"
_KAGGLE_REPO = f"/kaggle/working/{REPO_NAME}"


def repo_root() -> str:
    """Absolute path to the repository root on this platform."""
    if PLATFORM == "colab":
        return _COLAB_REPO
    if PLATFORM == "kaggle":
        return _KAGGLE_REPO
    # Local: this file lives at <root>/src/utils/environment.py
    return os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))


def runtime_root() -> str:
    """Directory where active (hot) experiment runs are written.

    Always fast local/runtime storage: the repo dir on local/Kaggle, and the
    repo dir under /content on Colab (never Drive - see plan section 24).
    """
    return repo_root()


def mirror_root() -> str:
    """Directory where artifacts must be copied to survive the session.

    Colab: the repo dir on Google Drive (mounted first if needed).
    Kaggle: identical to runtime_root - /kaggle/working is persisted to the
            user's Kaggle account as the notebook output automatically.
    Local: identical to runtime_root - nothing to mirror.
    """
    if PLATFORM == "colab":
        mount_drive()
        return os.path.join("/content/drive/MyDrive", REPO_NAME)
    return runtime_root()


def data_dir() -> str:
    """Where raw datasets and tokenized caches live.

    These files are regenerable (download + BPE encode), so on Colab they stay
    in runtime storage; only the small, non-regenerable tokenizer files get
    mirrored with each run (see sync_run_to_drive).
    """
    return os.path.join(runtime_root(), "data")


def runs_dir() -> str:
    """Top-level directory holding one subdirectory per training run."""
    return os.path.join(runtime_root(), "runs")


# --------------------------------------------------------------------------
# Colab Drive mount + mirroring
# --------------------------------------------------------------------------
def drive_mounted() -> bool:
    """True if /content/drive/MyDrive already exists."""
    return os.path.isdir("/content/drive/MyDrive")


def mount_drive() -> bool:
    """Mount Google Drive if we are on Colab and it is not mounted yet.

    Mounting opens an OAuth prompt; in a script context it will ask the user
    to click the auth link, which is the expected Colab behaviour.
    Returns True when /content/drive/MyDrive is usable afterwards.
    """
    if PLATFORM != "colab":
        return False
    if drive_mounted():
        return True
    try:
        from google.colab import drive
        drive.mount("/content/drive")
    except Exception:
        return False
    return drive_mounted()


_MIRROR_SMALL_NAMES = (
    "config.yaml", "environment.txt", "model_summary.txt",
    "metrics.csv", "train.log", "samples.txt", "tokenizer.json", "meta.json",
)


def sync_run_to_drive(run_dir: str, full: bool = False) -> bool:
    """Copy a run directory into the Drive mirror.

    WHAT
    ----
    Small artifacts (config, logs, metrics, samples, tokenizer) are copied on
    every call because they are tiny and cheap.
    Checkpoints (.pt) are copied only when full=True - they are large and
    Drive writes are slow, so they are mirrored at checkpoint intervals, not
    every evaluation.

    WHY
    ----
    Colab disconnects. If checkpoints only exist on /content they are gone.
    Mirroring periodically is the plan's "train local, copy to Drive" rule.

    Returns True if a mirror was needed and succeeded (or was unnecessary),
    False on mirror failure (never raises - I/O problems must not kill runs).
    """
    if PLATFORM != "colab":
        return True
    try:
        if not mount_drive():
            return False
        dst_root = os.path.join("/content/drive/MyDrive", REPO_NAME)
        dst = os.path.join(dst_root, os.path.relpath(run_dir, runs_dir()))
        os.makedirs(dst, exist_ok=True)
        for name in os.listdir(run_dir):
            if os.path.isdir(os.path.join(run_dir, name)):
                continue
            if name.endswith(".pt") and not full:
                continue
            shutil.copy2(os.path.join(run_dir, name), os.path.join(dst, name))
        return True
    except OSError:
        return False


# --------------------------------------------------------------------------
# Environment record (plan section 7: a run without config is not a result)
# --------------------------------------------------------------------------
def collect_environment() -> dict:
    """Gather everything needed to reproduce (or explain) a run."""
    info = {
        "platform": PLATFORM,
        "hostname": socket.gethostname(),
        "python_version": _platform.python_version(),
        "os": f"{_platform.system()} {_platform.release()}",
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    if torch is not None:
        info["torch_version"] = torch.__version__
        info["cuda_available"] = bool(torch.cuda.is_available())
        info["cuda_version"] = torch.version.cuda
        if torch.cuda.is_available():
            info["gpu_name"] = torch.cuda.get_device_name(0)
            info["gpu_vram_gb"] = round(
                torch.cuda.get_device_properties(0).total_memory / 1e9, 2)
            info["gpu_capability"] = torch.cuda.get_device_capability(0)
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            info["accelerator"] = "mps"
        else:
            info["accelerator"] = "cpu"
    info["git_commit"] = _git_commit()
    return info


def _git_commit() -> str:
    """Short git hash of the repo when available, else 'unknown'."""
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=repo_root(), capture_output=True, text=True, timeout=10)
        if out.returncode == 0:
            return out.stdout.strip()
    except (OSError, subprocess.SubprocessError):
        pass
    return "unknown"


def write_environment_file(run_dir: str) -> None:
    """Dump collect_environment() as environment.txt inside a run dir."""
    os.makedirs(run_dir, exist_ok=True)
    info = collect_environment()
    lines = [f"{key}: {value}" for key, value in sorted(info.items())]
    with open(os.path.join(run_dir, "environment.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def print_environment() -> None:
    """Human-readable startup banner (plan section 23)."""
    info = collect_environment()
    gpu = info.get("gpu_name", info.get("accelerator", "unknown"))
    vram = info.get("gpu_vram_gb", "n/a")
    print("=" * 72)
    print("nano-gpt-lab environment")
    print(f"  platform      : {info['platform']}")
    print(f"  accelerator   : {gpu} (VRAM {vram} GB)" if vram != "n/a"
          else f"  accelerator   : {gpu}")
    print(f"  torch         : {info.get('torch_version', 'n/a')}"
          f" | cuda {info.get('cuda_version', 'n/a')}")
    print(f"  python        : {info['python_version']}")
    print(f"  repo          : {repo_root()}")
    print(f"  git           : {info['git_commit']}")
    print("=" * 72)


def is_notebook() -> bool:
    """True when running inside an interactive notebook kernel.

    Used by scripts/notebooks to pick between printing and raising on errors.
    """
    try:
        shell = get_ipython().__class__.__name__  # type: ignore[name-defined]
        return shell == "ZMQInteractiveShell"
    except NameError:
        return False
