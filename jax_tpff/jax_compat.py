"""JAX 0.11 compatibility and local JAX-FEM path discovery."""

import os
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
_candidates = [
    Path(os.environ["JAX_FEM_PATH"]) if os.environ.get("JAX_FEM_PATH") else None,
    _ROOT / "source" / "jax-fem",
    _ROOT / "third_party" / "jax-fem",
    _ROOT.parent / "jax-cp",
]
for _candidate in _candidates:
    if _candidate is not None and (_candidate / "jax_fem").is_dir():
        _path = str(_candidate)
        if _path not in sys.path:
            sys.path.insert(0, _path)
        break

import jax

if not hasattr(jax, "tree_map"):
    jax.tree_map = jax.tree.map
