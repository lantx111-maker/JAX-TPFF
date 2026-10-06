#!/usr/bin/env python3
"""Run the thermal-shock refinement matrix and control cases."""

from __future__ import annotations

import logging
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".mplconfig"))
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("JAX_PLATFORMS", "cpu")
sys.path.insert(0, str(ROOT))

import jax_tpff.jax_compat  # noqa: F401,E402
from jax_tpff.cases import CASES  # noqa: E402
from jax_tpff.staggered import run_case  # noqa: E402


DEFAULT_NAMES = [
    "TS_100x040_n040_k0",
    "TS_100x040_n040_kd",
    "TS_100x040_n080_k0",
    "TS_100x040_n080_kd",
    "TS_200x080_n040_k0",
    "TS_200x080_n040_kd",
    "TS_200x080_n080_k0",
    "TS_200x080_n080_kd",
    "TS_400x160_n040_k0",
    "TS_400x160_n040_kd",
    "TS_Bi10_n080_k0",
    "TS_Bi10_n080_kd",
    "TS_free_n080_k0",
    "TS_free_n080_kd",
    "TS_gcT_n080_k0",
    "TS_gcT_n080_kd",
    "TS_AT1_n080_k0",
    "TS_AT1_n080_kd",
]


def main():
    logging.getLogger("jax_fem").setLevel(logging.WARNING)
    names = [arg for arg in sys.argv[1:] if not arg.startswith("-")] or DEFAULT_NAMES
    force = "--force" in sys.argv
    out = ROOT / "results/manuscript"
    for name in names:
        if name not in CASES:
            raise KeyError(f"Unknown manuscript case: {name}")
        summary = out / name / "summary.json"
        if summary.exists() and not force:
            print(f"=== skip {name} (summary exists) ===", flush=True)
            continue
        print(f"\n=== running {name} ===", flush=True)
        run_case(CASES[name], out)


if __name__ == "__main__":
    main()
