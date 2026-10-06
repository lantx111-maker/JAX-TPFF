#!/usr/bin/env python3
"""Run the CMAME-density layer (ℓ/h ≳ 4, tens of thousands of cells)."""
from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".mplconfig"))
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("JAX_PLATFORMS", "cpu")

import logging

logging.getLogger("jax_fem").setLevel(logging.WARNING)

import jax_tpff.jax_compat  # noqa: F401,E402
from jax_tpff.cases import CASES
from jax_tpff.dump_catalogue import write_campaign_index, write_catalogue
from jax_tpff.staggered import run_case

# Quench pair first (39k cells, 21 steps), then Miehe 256×256, then L-BFGS / AMR.
NAMES = [
    "T2_quench_cmame_kd",
    "T2_quench_cmame_k0",
    "T2_quench_pfczm_cmame",
    "M1_tension_cmame",
    "M1_shear_cmame",
    "T2_quench_lbfgs_cmame_kd",
    "T2_quench_lbfgs_cmame_k0",
    "M1_tension_lbfgs_cmame",
    "M1_shear_lbfgs_cmame",
    "M1_tension_amr_cmame",
    "T2_quench_amr_cmame",
    "T3_shock_cmame",
]


def main():
    out = ROOT / "outputs"
    force = "--force" in sys.argv
    only = [a for a in sys.argv[1:] if not a.startswith("-")]
    names = only or NAMES
    for name in names:
        summary = out / name / "summary.json"
        if summary.exists() and not force:
            print(f"=== skip {name} (summary exists) ===", flush=True)
            continue
        print(f"\n=== running {name} ===", flush=True)
        try:
            run_case(CASES[name], out)
        except Exception as exc:
            print(f"=== FAILED {name}: {type(exc).__name__}: {exc} ===", flush=True)
            import traceback

            traceback.print_exc()
            continue
        write_campaign_index(ROOT)
    write_catalogue(ROOT)
    write_campaign_index(ROOT)
    print("=== CMAME-density campaign finished ===", flush=True)


if __name__ == "__main__":
    main()
