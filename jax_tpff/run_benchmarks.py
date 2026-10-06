#!/usr/bin/env python3
"""Run four literature benchmark problems.

M3 three-point bending (Jin 2024), M4 Winkler L-panel PF-CZM (Jin 2024),
M5 notched plate with an eccentric hole (Ambati-type), and isothermal
PF-CZM shear on the Miehe square.
"""
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

NAMES = ["M1_shear_pfczm", "M5_hole", "M3_tpb", "M4_lshape"]


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
    print("=== benchmark campaign finished ===", flush=True)


if __name__ == "__main__":
    main()
