#!/usr/bin/env python3
"""Run a frozen JAX-TPFF case.

Example:
  python -m jax_tpff.run_case M1_tension_smoke
  python -m jax_tpff.run_case --list
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".mplconfig"))
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

sys.path.insert(0, str(ROOT))

import jax_tpff.jax_compat  # noqa: F401

from jax_tpff.cases import CASES, list_cases
from jax_tpff.dump_catalogue import write_campaign_index, write_catalogue
from jax_tpff.staggered import run_case


def main():
    p = argparse.ArgumentParser()
    p.add_argument("name", nargs="?", help="case name")
    p.add_argument("--list", action="store_true")
    p.add_argument("--tag", default=None)
    p.add_argument("--all", action="store_true", help="run every registered case, including production")
    p.add_argument("--force", action="store_true", help="re-run even if summary.json exists")
    p.add_argument("--out", default=str(ROOT / "outputs"))
    args = p.parse_args()
    logging.getLogger("jax_fem").setLevel(logging.WARNING)

    if args.list:
        for c in list_cases(args.tag):
            print(f"{c.name:24s}  {c.mode:8s}  {c.Nx}x{c.Ny}  ell={c.mat.ell}  {c.description}")
        return

    names = []
    if args.name:
        names = [args.name]
    elif args.all:
        names = [c.name for c in list_cases(args.tag)]
    elif args.tag:
        names = [c.name for c in list_cases(args.tag) if "production" not in c.tags]
    else:
        raise SystemExit("pass a case name, --all, --tag, or --list")

    out = Path(args.out)
    for name in names:
        if name not in CASES:
            raise SystemExit(f"unknown case {name}. Use --list.")
        summary = out / name / "summary.json"
        if summary.exists() and not args.force:
            print(f"=== skip {name} (summary exists) ===", flush=True)
            continue
        print(f"\n=== running {name} ===", flush=True)
        run_case(CASES[name], out)
        write_campaign_index(ROOT)

    write_catalogue(ROOT)
    write_campaign_index(ROOT)


if __name__ == "__main__":
    main()
