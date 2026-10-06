#!/usr/bin/env python3
"""Run the selected reference-quality CPU cases."""
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

# Run lower-cost cases before the 80x80 L-BFGS calculations.
NAMES = [
    "T2_quench_lbfgs_k0gct",
    "T2_quench_lbfgs_kd10",
    "T2_quench_pfczm_ref",
    "T3_shock_lbfgs_mid",
    "I1_gcT_inverse_cpu",
    "M1_tension_at1_ref",
    "T1_thermo_shear_ref",
    "M1_shear_ref_ext",
    "M1_tension_lbfgs_ref",
    "M1_shear_lbfgs_ref",
]


def main():
    out = ROOT / "outputs"
    force = "--force" in sys.argv
    for name in NAMES:
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
    print("=== reference case suite finished ===", flush=True)


if __name__ == "__main__":
    main()
