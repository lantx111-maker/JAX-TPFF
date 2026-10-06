#!/usr/bin/env python3
"""Run the selected CPU case suite."""
from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".mplconfig"))
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import logging

logging.getLogger("jax_fem").setLevel(logging.WARNING)

import jax_tpff.jax_compat  # noqa: F401,E402
from jax_tpff.cases import CASES
from jax_tpff.dump_catalogue import write_campaign_index, write_catalogue
from jax_tpff.staggered import run_case

NAMES = [
    "M1_tension_lbfgs",
    "M1_shear_lbfgs",
    "M2_cyclic_lbfgsb",
    "T2_quench_lbfgs_k0",
    "T2_quench_lbfgs_kd",
    "T2_quench_pfczm_lbfgs",
    "T2_quench_reheat_lbfgsb",
    "T3_shock_lbfgs",
    "I1_gcT_inverse",
]


def main():
    out = ROOT / "outputs"
    for name in NAMES:
        summary = out / name / "summary.json"
        if summary.exists():
            print(f"=== skip {name} (summary exists) ===", flush=True)
            continue
        print(f"\n=== running {name} ===", flush=True)
        try:
            run_case(CASES[name], out)
        except Exception as exc:
            print(f"=== FAILED {name}: {type(exc).__name__}: {exc} ===", flush=True)
            continue
        write_campaign_index(ROOT)
    write_catalogue(ROOT)
    write_campaign_index(ROOT)
    print("=== selected case suite finished ===", flush=True)


if __name__ == "__main__":
    main()
