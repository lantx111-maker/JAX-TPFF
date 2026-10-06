#!/usr/bin/env python3
"""Run the supplementary multi-case study."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".mplconfig"))
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

from jax_tpff.cases import CASES
from jax_tpff.staggered import run_case


CASES_EXPANSION = [
    "T2_mesh_050x020_k0", "T2_mesh_200x080_k0", "T2_mesh_312x125_k0",
    "T2_length_l1p0_k0", "T2_length_l1p0_kd",
    "T2_length_l0p75_k0", "T2_length_l0p75_kd",
    "T2_length_l0p25_k0", "T2_length_l0p25_kd",
    "T2_pfczm_p0_k0", "T2_pfczm_p0_kd",
    "T2_pfczm_ft0p5_k0", "T2_pfczm_ft0p5_kd",
    "T2_convective_Bi0p1", "T2_convective_Bi1p0", "T2_convective_Bi10p0",
    "T2_notched_p0", "T2_reheat_p0", "T2_amr_p0", "T3_resolved_path",
    "T2_kres_1em02", "T2_kres_1em03",
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("names", nargs="*")
    ap.add_argument("--out", default=str(ROOT / "outputs_expansion"))
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    out = Path(args.out)
    for name in args.names or CASES_EXPANSION:
        summary = out / name / "summary.json"
        if summary.exists() and not args.force:
            print(f"skip {name}", flush=True)
            continue
        print(f"=== {name} ===", flush=True)
        run_case(CASES[name], out)


if __name__ == "__main__":
    main()
