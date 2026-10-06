#!/usr/bin/env python3
"""Validate completeness and acceptance criteria for manuscript cases."""

from __future__ import annotations

import csv
import json
from pathlib import Path

from jax_tpff.cases import CASES
from jax_tpff.run_manuscript import DEFAULT_NAMES


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/manuscript"


def _truth(value: str) -> bool:
    return value.strip().lower() == "true"


def main() -> None:
    report = []
    failures = []
    for name in DEFAULT_NAMES:
        case = CASES[name]
        history_path = OUT / name / "history.csv"
        summary_path = OUT / name / "summary.json"
        if not history_path.exists() or not summary_path.exists():
            failures.append(f"{name}: missing history or summary")
            continue
        with history_path.open(newline="") as stream:
            rows = list(csv.DictReader(stream))
        expected = len(case.times) - 1
        max_kkt = max(abs(float(row["damage_kkt_residual"])) for row in rows)
        min_dd = min(float(row["min_damage_increment"]) for row in rows)
        max_balance = max(abs(float(row["heat_balance_relative"])) for row in rows)
        accepted = (
            len(rows) == expected
            and all(_truth(row["converged"]) for row in rows)
            and all(_truth(row["damage_active_set_converged"]) for row in rows)
            and min_dd >= -1.0e-12
            and max_kkt <= 1.0e-10
            and min(float(row["T_min"]) for row in rows) >= case.T_cold - 1.0e-8
            and max(float(row["T_max"]) for row in rows) <= case.mat.T_ref + 1.0e-6
        )
        if not accepted:
            failures.append(f"{name}: acceptance criterion failed")
        report.append(
            {
                "case": name,
                "steps": len(rows),
                "expected_steps": expected,
                "accepted": accepted,
                "max_kkt": max_kkt,
                "min_damage_increment": min_dd,
                "max_heat_balance_relative": max_balance,
            }
        )
    print(json.dumps(report, indent=2))
    if failures:
        raise SystemExit("\n".join(failures))


if __name__ == "__main__":
    main()
