"""pytest bootstrap: make `src` importable from tests/."""
import os
import sys

# Windows/conda OpenMP duplicate-runtime workaround (see scripts/train.py).
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
