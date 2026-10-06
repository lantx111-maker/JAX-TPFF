#!/usr/bin/env python3
"""One-dimensional transient-conduction verification on a 2-D FE strip."""

from __future__ import annotations

import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".mplconfig"))
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import jax.numpy as jnp
from jax_fem.generate_mesh import Mesh, rectangle_mesh
from jax_fem.solver import solver
from jax_fem.utils import save_sol

from jax_tpff.constitutive import Material
from jax_tpff.problems import Heat


def exact_theta(x, t, L, diffusivity, terms=300):
    n = np.arange(terms)[:, None]
    beta = (2*n+1)*np.pi/(2*L)
    return np.sum(4/((2*n+1)*np.pi)*np.sin(beta*x[None,:])*np.exp(-diffusivity*beta**2*t), axis=0)


def solve_strip(nx, nstep, mat, L, Ly, T0, Tc, t_end, lumped=True):
    ny = max(1, nx // 20)
    raw = rectangle_mesh(Nx=nx, Ny=ny, domain_x=L, domain_y=Ly)
    mesh = Mesh(raw.points, raw.cells_dict["quad"], ele_type="QUAD4")

    def left(p):
        return jnp.isclose(p[0], 0.0, atol=1e-10)

    def cold(_p):
        return Tc

    problem = Heat(
        mesh,
        vec=1,
        dim=2,
        ele_type="QUAD4",
        dirichlet_bc_info=[[left], [0], [cold]],
        additional_info=(mat, 0.0, Tc, lumped),
    )
    sol = [T0 * np.ones((len(mesh.points), 1))]
    sol[0][np.isclose(mesh.points[:, 0], 0.0), 0] = Tc
    dt = t_end / nstep
    for _ in range(nstep):
        old = sol[0].copy()
        problem.set_params([old, dt, None])
        sol = solver(problem, solver_options={"umfpack_solver": {}})
    x = mesh.points[:, 0]
    theta_exact = exact_theta(x, t_end, L, mat.kappa0 / mat.heat_capacity)
    T_exact = Tc + (T0 - Tc) * theta_exact
    err = sol[0][:, 0] - T_exact
    rel_l2 = float(np.linalg.norm(err) / np.linalg.norm(T_exact - Tc))
    linf = float(np.max(np.abs(err)))
    return mesh, problem, sol, T_exact, err, rel_l2, linf


def main():
    mpl.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
            "mathtext.fontset": "stix",
            "font.size": 10,
            "axes.labelsize": 10.5,
            "axes.linewidth": 0.8,
            "axes.spines.top": True,
            "axes.spines.right": True,
            "xtick.direction": "in",
            "ytick.direction": "in",
            "xtick.top": False,
            "ytick.right": False,
            "legend.frameon": False,
            "pdf.fonttype": 42,
        }
    )
    out = ROOT / "results" / "verification" / "heat_verification"
    (out/"vtk").mkdir(parents=True, exist_ok=True)
    L, Ly, nx = 12.5, 0.625, 80
    T0, Tc, t_end, nstep = 573.15, 293.15, 0.05, 50
    mat = Material(E=1.0, nu=0.2, Gc0=1.0, ell=1.0, heat_capacity=3.5024e-3, kappa0=0.031)
    mesh, problem, sol, T_exact, err, rel_l2, linf = solve_strip(
        nx, nstep, mat, L, Ly, T0, Tc, t_end
    )
    consistent = solve_strip(nx, nstep, mat, L, Ly, T0, Tc, t_end, lumped=False)
    x = mesh.points[:,0]
    order = np.argsort(x)

    spatial = []
    for nxi in (20, 40, 80, 160):
        result = solve_strip(nxi, 400, mat, L, Ly, T0, Tc, t_end)
        spatial.append({"Nx": nxi, "h_mm": L/nxi, "relative_L2": result[-2], "Linf_K": result[-1]})
    temporal = []
    for ns in (10, 20, 40, 80, 160):
        result = solve_strip(160, ns, mat, L, Ly, T0, Tc, t_end)
        temporal.append({"steps": ns, "dt_s": t_end/ns, "relative_L2": result[-2], "Linf_K": result[-1]})

    fig, axs = plt.subplots(1, 3, figsize=(7.15, 2.85))
    axs[0].plot(x[order],T_exact[order],"k-",lw=1.4,label="series solution")
    axs[0].plot(x[order],sol[0][:,0][order],"o",ms=1.8,mfc="none",label="lumped Q1")
    axs[0].set(xlabel="x [mm]",ylabel="T [K]")
    axs[0].legend(loc="lower right")
    nx_values = [r["Nx"] for r in spatial]
    nt_values = [r["steps"] for r in temporal]
    axs[1].plot(nx_values, [r["relative_L2"] for r in spatial], "o-")
    h_ref = np.asarray(nx_values, dtype=float)
    ref_space = spatial[0]["relative_L2"] * (h_ref[0] / h_ref) ** 2
    axs[1].plot(nx_values, ref_space, "k--", lw=0.8, label=r"$O(h^2)$")
    axs[1].set_xscale("log", base=2)
    axs[1].set_yscale("log")
    axs[1].set_xticks(nx_values, [str(v) for v in nx_values])
    axs[1].set(xlabel=r"$N_x$",ylabel="relative $L_2$ error")
    axs[1].legend(loc="lower left")
    axs[2].plot(nt_values, [r["relative_L2"] for r in temporal], "o-")
    nt_ref = np.asarray(nt_values, dtype=float)
    ref_time = temporal[0]["relative_L2"] * nt_ref[0] / nt_ref
    axs[2].plot(nt_values, ref_time, "k--", lw=0.8, label=r"$O(\Delta t)$")
    axs[2].set_xscale("log", base=2)
    axs[2].set_yscale("log")
    axs[2].set_xticks(nt_values, [str(v) for v in nt_values])
    axs[2].set(xlabel=r"$N_t$",ylabel="relative $L_2$ error")
    axs[2].legend(loc="lower left")
    for label, ax in zip(("(a)", "(b)", "(c)"), axs):
        ax.xaxis.set_label_coords(0.5, -0.14)
        ax.text(
            0.5, -0.34, label,
            transform=ax.transAxes,
            ha="center", va="top",
            fontweight="bold", clip_on=False,
        )
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_linewidth(0.8)
        ax.minorticks_on()
        ax.tick_params(axis="both", which="major", direction="in", top=False, right=False, width=0.8, length=4.0)
        ax.tick_params(axis="both", which="minor", direction="in", top=False, right=False, width=0.7, length=2.2)
        ax.grid(alpha=.22, linestyle=":", linewidth=.45)
    fig.tight_layout(rect=(0.0, 0.04, 1.0, 1.0), w_pad=1.3)
    fig.savefig(out/"heat_verification.pdf", bbox_inches="tight", pad_inches=0.03)
    (ROOT/"paper_artifacts"/"figures").mkdir(parents=True, exist_ok=True)
    fig.savefig(ROOT/"paper_artifacts"/"figures"/"heat_verification.pdf", bbox_inches="tight", pad_inches=0.03)
    plt.close(fig)
    save_sol(problem.fes[0],sol[0],str(out/"vtk"/"terminal.vtu"),point_infos=[("T",sol[0]),("T_exact",T_exact[:,None]),("error",err[:,None])])
    data={
        "case":"heat_verification",
        "Nx":nx,
        "Ny":max(1,nx//20),
        "n_steps":nstep,
        "t_end":t_end,
        "relative_L2":rel_l2,
        "Linf_K":linf,
        "T_min":float(sol[0].min()),
        "T_max":float(sol[0].max()),
        "spatial":spatial,
        "temporal":temporal,
        "mass_matrix":"row-sum lumped",
        "consistent_mass_comparison":{
            "relative_L2":consistent[-2],
            "Linf_K":consistent[-1],
            "T_min":float(consistent[2][0].min()),
            "T_max":float(consistent[2][0].max()),
        },
    }
    (out/"summary.json").write_text(json.dumps(data,indent=2))
    print(json.dumps(data,indent=2))


if __name__ == "__main__":
    main()
