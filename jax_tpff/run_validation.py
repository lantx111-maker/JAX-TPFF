#!/usr/bin/env python3
"""Run the verification, solver-comparison and temporal-refinement cases."""

from __future__ import annotations

import logging
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".mplconfig"))
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
sys.path.insert(0, str(ROOT))

import jax_tpff.jax_compat  # noqa: F401

from jax_tpff.cases import CASES
from jax_tpff.staggered import run_case


NAMES = [
    "patch_staggered",
    "patch_jax",
    "patch_lbfgs",
    "patch_lbfgsb",
    "T2_time_005",
    "T2_time_020",
    "T2_time_040",
    "T2_time_080",
    "T2_time_020_kd",
    "T2_time_040_kd",
]


def main():
    logging.getLogger("jax_fem").setLevel(logging.WARNING)
    out = ROOT / "results/verification"
    for name in NAMES:
        summary = out / name / "summary.json"
        if summary.exists():
            print(f"=== skip {name} (summary exists) ===", flush=True)
            continue
        print(f"\n=== running {name} ===", flush=True)
        run_case(CASES[name], out)


if __name__ == "__main__":
    main()
