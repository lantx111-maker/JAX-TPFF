#!/usr/bin/env python3
"""Generate publication plots for the supplementary case study."""

from __future__ import annotations

import csv
import json
import os
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
os.environ["MPLCONFIGDIR"] = str(ROOT / ".mplconfig")

import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np

FIG = ROOT / "paper_artifacts" / "figures"
FIG.mkdir(exist_ok=True)


def summary(root, name):
    return json.loads((ROOT/root/name/"summary.json").read_text())


def history(root, name):
    with (ROOT/root/name/"history.csv").open() as f:
        rows=list(csv.DictReader(f))
    return {k:np.array([float(r[k]) for r in rows]) for k in rows[0] if k not in ("converged",)}


def save(fig, name):
    for ax in fig.axes:
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_linewidth(0.8)
        ax.minorticks_on()
        ax.tick_params(axis="both", which="major", direction="in", top=False, right=False, width=0.8, length=3.8)
        ax.tick_params(axis="both", which="minor", direction="in", top=False, right=False, width=0.7, length=2.1)
    fig.tight_layout(pad=0.45, w_pad=0.75)
    fig.savefig(FIG/name, bbox_inches="tight", pad_inches=0.03)
    plt.close(fig)


def main():
    mpl.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
            "mathtext.fontset": "stix",
            "font.size": 9.2,
            "axes.labelsize": 9.6,
            "axes.linewidth": 0.8,
            "axes.spines.top": True,
            "axes.spines.right": True,
            "axes.grid": True,
            "grid.alpha": 0.22,
            "grid.linestyle": ":",
            "grid.linewidth": 0.45,
            "xtick.direction": "in",
            "ytick.direction": "in",
            "xtick.top": False,
            "ytick.right": False,
            "legend.frameon": False,
            "legend.fontsize": 7.8,
            "lines.linewidth": 1.35,
            "lines.markersize": 4.2,
            "pdf.fonttype": 42,
        }
    )
    shutil.copyfile(ROOT/"outputs_expansion"/"heat_verification"/"heat_verification.pdf", FIG/"exp_heat_verification.pdf")

    # Fixed-length mesh-density convergence.
    mesh_rows = [
        summary("outputs_expansion", "T2_mesh_050x020_k0"),
        summary("outputs_p0", "T2_p0_at2_k0"),
        summary("outputs_expansion", "T2_mesh_200x080_k0"),
        summary("outputs_expansion", "T2_mesh_312x125_k0"),
    ]
    cells = np.array([r["n_cell_final"] for r in mesh_rows])
    last = [r["last"] for r in mesh_rows]
    fig, axs = plt.subplots(1, 3, figsize=(7.2, 2.5))
    axs[0].semilogx(cells, [r["T_core_mean"] for r in last], "o-")
    axs[1].semilogx(cells, [r["fracture_surface"] for r in last], "o-")
    axs[2].semilogx(cells, [r["damage_area_d08"] for r in last], "o-")
    for ax in axs:
        ax.set_xlabel("QUAD4 cells")
        ax.axvline(4000, color=".45", ls="--", lw=.9)
    axs[0].set_ylabel(r"$\overline{T}_{core}$ [K]")
    axs[1].set_ylabel(r"$\Gamma_\ell$ [mm]")
    axs[2].set_ylabel(r"$A_{d\geq0.8}$ [mm$^2$]")
    save(fig, "exp_mesh_density.pdf")

    # Length scale at fixed ell/h approximately four.
    ell=np.array([1.0,.75,.5])
    k0=[summary("outputs_expansion","T2_length_l1p0_k0")["last"],summary("outputs_expansion","T2_length_l0p75_k0")["last"],summary("outputs_p0","T2_p0_at2_k0")["last"]]
    kd=[summary("outputs_expansion","T2_length_l1p0_kd")["last"],summary("outputs_expansion","T2_length_l0p75_kd")["last"],summary("outputs_p0","T2_p0_at2_kd")["last"]]
    fig,axs=plt.subplots(1,3,figsize=(7.2,2.5))
    axs[0].plot(ell,[r["fracture_surface"] for r in k0],"o-",label=r"$\kappa_0$")
    axs[0].plot(ell,[r["fracture_surface"] for r in kd],"s--",label=r"$\kappa(d)$")
    axs[0].set(xlabel=r"$\ell$ [mm]",ylabel=r"$\Gamma_\ell$ [mm]"); axs[0].legend(loc="lower center", bbox_to_anchor=(0.5,1.01), ncol=2)
    axs[1].plot(ell,[r["damage_area_d08"] for r in k0],"o-")
    axs[1].plot(ell,[r["damage_area_d08"] for r in kd],"s--")
    axs[1].set(xlabel=r"$\ell$ [mm]",ylabel=r"$A_{d\geq0.8}$ [mm$^2$]")
    axs[2].plot(ell,[r["T_core_mean"] for r in k0],"o-")
    axs[2].plot(ell,[r["T_core_mean"] for r in kd],"s--")
    axs[2].set(xlabel=r"$\ell$ [mm]",ylabel=r"$\overline{T}_{core}$ [K]")
    save(fig,"exp_length_scale.pdf")

    # Convection histories.
    fig,axs=plt.subplots(1,3,figsize=(7.2,2.5))
    for name,label,ls in [("T2_convective_Bi0p1","Bi=0.1","-"),("T2_convective_Bi1p0","Bi=1","-"),("T2_convective_Bi10p0","Bi=10","--")]:
        h=history("outputs_expansion",name)
        axs[0].plot(h["time"],h["T_core_mean"],ls,label=label)
        axs[1].plot(h["time"],h["d_max"],ls)
        axs[2].plot(h["time"],h["heat_outflow"],ls)
    axs[0].set(xlabel="t [s]",ylabel=r"$\overline{T}_{core}$ [K]"); axs[0].legend(loc="lower center", bbox_to_anchor=(0.5,1.01), ncol=3)
    axs[1].set(xlabel="t [s]",ylabel=r"$d_{max}$")
    axs[2].set(xlabel="t [s]",ylabel="heat outflow [model units]")
    save(fig,"exp_convection.pdf")

    # Quench-reheat irreversibility.
    h=history("outputs_expansion","T2_reheat_p0")
    fig,axs=plt.subplots(1,3,figsize=(7.2,2.5))
    axs[0].plot(h["time"],h["T_core_mean"],"o-")
    axs[1].plot(h["time"],h["fracture_surface"],"o-")
    axs[2].plot(h["time"],h["damage_area_d08"],"o-")
    for ax in axs: ax.axvline(.06,color=".35",ls="--",lw=1)
    axs[0].set(xlabel="t [s]",ylabel=r"$\overline{T}_{core}$ [K]")
    axs[1].set(xlabel="t [s]",ylabel=r"$\Gamma_\ell$ [mm]")
    axs[2].set(xlabel="t [s]",ylabel=r"$A_{d\geq0.8}$ [mm$^2$]")
    save(fig,"exp_reheat.pdf")

    # AMR versus uniform mesh.
    ha=history("outputs_expansion","T2_amr_p0")
    hu=history("outputs_p0","T2_p0_at2_k0")
    fig,axs=plt.subplots(1,3,figsize=(7.2,2.5))
    axs[0].plot(ha["time"],ha["n_cell"],"o-")
    axs[1].plot(hu["time"],hu["T_core_mean"],label="uniform QUAD4")
    axs[1].plot(ha["time"],ha["T_core_mean"],"--",label="AMR TRI3")
    axs[2].plot(hu["time"],hu["fracture_surface"],label="uniform QUAD4")
    axs[2].plot(ha["time"],ha["fracture_surface"],"--",label="AMR TRI3")
    axs[0].set(xlabel="t [s]",ylabel="active cells")
    axs[1].set(xlabel="t [s]",ylabel=r"$\overline{T}_{core}$ [K]"); axs[1].legend(loc="lower center", bbox_to_anchor=(0.5,1.01), ncol=2)
    axs[2].set(xlabel="t [s]",ylabel=r"$\Gamma_\ell$ [mm]")
    save(fig,"exp_amr_compare.pdf")

    # Residual conductivity factor design.
    kres=np.array([1.,1e-2,1e-3,1e-4])
    rows=[summary("outputs_p0","T2_p0_at2_k0")["last"],summary("outputs_expansion","T2_kres_1em02")["last"],summary("outputs_expansion","T2_kres_1em03")["last"],summary("outputs_p0","T2_p0_at2_kd")["last"]]
    fig,axs=plt.subplots(1,3,figsize=(7.2,2.5))
    axs[0].semilogx(kres,[r["T_core_mean"] for r in rows],"o-")
    axs[1].semilogx(kres,[r["fracture_surface"] for r in rows],"o-")
    axs[2].semilogx(kres,[r["damage_area_d08"] for r in rows],"o-")
    for ax in axs: ax.invert_xaxis(); ax.set_xlabel(r"$k_\kappa$")
    axs[0].set_ylabel(r"$\overline{T}_{core}$ [K]")
    axs[1].set_ylabel(r"$\Gamma_\ell$ [mm]")
    axs[2].set_ylabel(r"$A_{d\geq0.8}$ [mm$^2$]")
    save(fig,"exp_kres_factor.pdf")


if __name__ == "__main__":
    main()
