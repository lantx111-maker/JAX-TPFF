"""Dump the frozen case catalogue and campaign index."""

from __future__ import annotations

import json
from pathlib import Path

from jax_tpff.cases import CASES


def write_catalogue(root: Path | None = None) -> Path:
    root = root or Path(__file__).resolve().parents[1]
    out = root / "outputs" / "case_catalogue.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = []
    for c in CASES.values():
        payload.append(
            {
                "name": c.name,
                "mode": c.mode,
                "Nx": c.Nx,
                "Ny": c.Ny,
                "Lx": c.Lx,
                "Ly": c.Ly,
                "ell": c.mat.ell,
                "h": c.Lx / c.Nx,
                "h_over_ell": (c.Lx / c.Nx) / c.mat.ell,
                "Gc0": c.mat.Gc0,
                "E": c.mat.E,
                "nu": c.mat.nu,
                "model": c.mat.model,
                "plane_stress": c.mat.plane_stress,
                "n_load": int(len(c.times) - 1 if c.mode in ("quench", "quench3d", "inverse") and c.times is not None else len(c.disps)),
                "solver": c.solver,
                "Nz": c.Nz,
                "max_inner": c.max_inner,
                "tags": list(c.tags),
                "description": c.description,
                "b1": c.mat.b1,
                "b2": c.mat.b2,
                "amr": c.amr,
                "degrade_conductivity": c.mat.degrade_conductivity,
            }
        )
    out.write_text(json.dumps(payload, indent=2))
    print(f"wrote {out} ({len(payload)} cases)")
    return out


def write_campaign_index(root: Path | None = None) -> Path:
    root = root or Path(__file__).resolve().parents[1]
    out_root = root / "outputs"
    cases = []
    for name, spec in CASES.items():
        summary_path = out_root / name / "summary.json"
        last = {}
        wall = None
        peak = None
        done = summary_path.exists()
        if done:
            data = json.loads(summary_path.read_text())
            last = data.get("last", {})
            wall = data.get("wall_s")
            peak = data.get("peak_force")
            n_node = last.get("n_node")
        else:
            if spec.mode == "quench3d":
                n_node = (spec.Nx + 1) * (spec.Ny + 1) * (spec.Nz + 1)
            else:
                n_node = (spec.Nx + 1) * (spec.Ny + 1)
        if spec.mode == "quench3d":
            ncomp = 5
        elif spec.mode in ("quench", "inverse") or spec.T_top is not None:
            ncomp = 4
        else:
            ncomp = 3
        cases.append(
            {
                "case": name,
                "mode": spec.mode,
                "Nx": spec.Nx,
                "Ny": spec.Ny,
                "ell": spec.mat.ell,
                "amr": spec.amr,
                "model": spec.mat.model,
                "solver": spec.solver,
                "n_node": n_node,
                "n_dof": None if n_node is None else int(n_node) * ncomp,
                "done": done,
                "wall_s": wall,
                "peak_force": peak,
                "n_steps": last.get("step"),
                "n_cell_final": last.get("n_cell"),
                "last": last,
            }
        )
    payload = {
        "backend": "jax-cpu",
        "n_registered": len(CASES),
        "n_done": sum(1 for c in cases if c["done"]),
        "missing": [c["case"] for c in cases if not c["done"]],
        "cases": cases,
    }
    out = out_root / "campaign_index.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2))
    print(f"wrote {out} ({payload['n_done']}/{payload['n_registered']} done)")
    return out


def main():
    write_catalogue()
    write_campaign_index()


if __name__ == "__main__":
    main()
