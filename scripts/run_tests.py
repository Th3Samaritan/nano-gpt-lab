"""run_tests.py - zero-dependency test runner for Colab/Kaggle/CI.

WHAT
----
Runs every test_* function in tests/*.py without needing pytest installed
(Kaggle base images often lack it). Prints one line per test and exits
non-zero on any failure. pytest remains fully supported (see pyproject).

HOW
----
Test files are loaded BY FILE PATH (importlib spec loader) rather than as
`tests.test_x` imports: a third-party PyPI package named `tests` exists
and can shadow the namespace package on some machines, so path loading is
the bulletproof choice everywhere.

WHERE
----
    python scripts/run_tests.py        # from the repo root
"""
from __future__ import annotations

import importlib.util
import os
import sys
import traceback

# Windows/conda OpenMP duplicate-runtime workaround (see scripts/train.py).
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, REPO_ROOT)  # makes `src` importable for the test files

MODULES = [
    "test_attention", "test_rope", "test_tokenizer",
    "test_transformer", "test_ffn", "test_optimizers", "test_logging",
    "test_metrics", "test_shots", "test_overfit",
]


def _load_by_path(name: str):
    """Import tests/<name>.py as an independent module (no package import)."""
    path = os.path.join(REPO_ROOT, "tests", f"{name}.py")
    spec = importlib.util.spec_from_file_location(f"nanolab_tests.{name}", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main() -> int:
    failures: list[str] = []
    total = 0
    for name in MODULES:
        module = _load_by_path(name)
        tests = [(n, getattr(module, n)) for n in sorted(dir(module))
                 if n.startswith("test_") and callable(getattr(module, n))]
        for tname, fn in tests:
            total += 1
            try:
                fn()
                print(f"PASS {name}.{tname}")
            except Exception:
                failures.append(f"{name}.{tname}")
                print(f"FAIL {name}.{tname}")
                traceback.print_exc()
    print("-" * 60)
    print(f"{total - len(failures)}/{total} tests passed")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
