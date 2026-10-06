#!/usr/bin/env python3
"""Generate publication curves from completed case CSV files."""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parents[1] / "figures"
OUT.mkdir(parents=True, exist_ok=True)

mpl.rcParams.update(
    {
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
        "mathtext.fontset": "stix",
        "font.size": 10,
        "axes.labelsize": 11,
        "axes.linewidth": 0.8,
        "axes.spines.top": True,
        "axes.spines.right": True,
        "xtick.direction": "in",
        "ytick.direction": "in",
        "xtick.top": False,
        "ytick.right": False,
        "legend.frameon": False,
        "legend.fontsize": 8.5,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.03,
        "pdf.fonttype": 42,
    }
)


def load(case: str):
    path = ROOT / "outputs" / case / "history.csv"
    with path.open() as f:
        rows = list(csv.DictReader(f))
    return rows


def col(rows, key):
    return np.array([float(r[key]) for r in rows if r.get(key) not in (None, "")])


def style_ax(ax):
    ax.grid(True, which="major", ls=":", lw=0.4, alpha=0.7)
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(0.8)
    ax.minorticks_on()
    ax.tick_params(axis="both", which="major", direction="in", top=False, right=False, width=0.8, length=3.5)
    ax.tick_params(axis="both", which="minor", direction="in", top=False, right=False, width=0.7, length=2.0)


def fig_m1_tension():
    fig, ax = plt.subplots(figsize=(3.45, 2.65))
    series = [
        ("M1_tension_cpu", "Staggered $40\\times 40$", "o", "#1f4e79"),
        ("M1_tension_ref", "Staggered $80\\times 80$", "s", "#2a9d8f"),
        ("M1_tension_lbfgs", "L-BFGS $40\\times 40$", "^", "#c0392b"),
        ("M1_tension_lbfgs_ref", "L-BFGS $80\\times 80$", "P", "#922b21"),
        ("M1_tension_at1_cpu", "AT1 staggered $40\\times 40$", "D", "#7f8c8d"),
        ("M1_tension_amr", "AMR TRI3", "v", "#8e44ad"),
        ("M1_tension_cmame", r"Staggered $256\times 256$", "x", "#1a5276"),
        ("M1_tension_amr_cmame", r"AMR, $50\mathrm{k}$ TRI3", "v", "#6c3483"),
    ]
    for case, lab, mk, c in series:
        r = load(case)
        ax.plot(col(r, "disp") * 1e3, col(r, "force"), "-" + mk, color=c, ms=3.2, lw=1.15, label=lab, markevery=max(len(r) // 10, 1))
    ax.set_xlabel(r"Displacement $u$ [$\times 10^{-3}$ mm]")
    ax.set_ylabel(r"Reaction $F$ [kN]")
    ax.legend(loc="upper left", handlelength=1.6)
    style_ax(ax)
    fig.savefig(OUT / "m1_tension_ld.pdf")
    plt.close(fig)


def fig_m1_shear():
    fig, ax = plt.subplots(figsize=(3.45, 2.65))
    series = [
        ("M1_shear_cpu", "Staggered $40\\times 40$", "o", "#1f4e79"),
        ("M1_shear_ref_ext", r"Staggered $80\times 80$ to $0.020$", "s", "#2a9d8f"),
        ("M1_shear_lbfgs", "L-BFGS $40\\times 40$", "^", "#c0392b"),
        ("M1_shear_lbfgs_ref", "L-BFGS $80\\times 80$", "P", "#922b21"),
        ("M1_shear_amr", "AMR TRI3", "v", "#8e44ad"),
        ("M1_shear_cmame", r"Staggered $256\times 256$", "x", "#1a5276"),
    ]
    for case, lab, mk, c in series:
        r = load(case)
        ax.plot(col(r, "disp"), col(r, "force"), "-" + mk, color=c, ms=3.2, lw=1.15, label=lab, markevery=max(len(r) // 8, 1))
    ax.set_xlabel(r"Displacement $u$ [mm]")
    ax.set_ylabel(r"Reaction $F$ [kN]")
    ax.legend(loc="lower right", handlelength=1.6)
    style_ax(ax)
    fig.savefig(OUT / "m1_shear_ld.pdf")
    plt.close(fig)


def fig_m2():
    fig, ax = plt.subplots(figsize=(3.45, 2.65))
    for case, lab, c in (
        ("M2_cyclic_cpu", "History-field staggered", "#1f4e79"),
        ("M2_cyclic_lbfgsb", "L-BFGS-B box", "#c0392b"),
    ):
        r = load(case)
        u, f = col(r, "disp"), col(r, "force")
        i_pk = int(np.argmax(u[: max(len(u) // 2, 1)]))
        i_un = i_pk + int(np.argmin(u[i_pk:]))
        ax.plot(u[: i_pk + 1], f[: i_pk + 1], "-", color=c, lw=1.45, label=lab, zorder=3)
        ax.plot(u[i_pk : i_un + 1], f[i_pk : i_un + 1], "--", color=c, lw=1.5, zorder=4)
        ax.plot(u[i_un:], f[i_un:], "-", color=c, lw=1.45, zorder=3)
        ax.scatter(u[i_pk], f[i_pk], s=32, marker="o", color=c, zorder=6, edgecolors="white", linewidths=0.5)
        ax.scatter(u[i_un], f[i_un], s=26, marker="s", color=c, zorder=6, edgecolors="white", linewidths=0.5)
    ax.annotate(
        "unload",
        xy=(0.0062, 0.20),
        fontsize=8,
        color="#444444",
        ha="center",
    )
    ax.set_xlabel(r"Displacement $u$ [mm]")
    ax.set_ylabel(r"Reaction $F$ [kN]")
    ax.legend(loc="upper left")
    style_ax(ax)
    fig.savefig(OUT / "m2_cyclic_ld.pdf")
    plt.close(fig)


def fig_t2_ncr():
    fig, ax = plt.subplots(figsize=(3.45, 2.65))
    series = [
        ("T2_quench_cpu", r"Staggered $\kappa_0$, $G_c(T)$", "#1f4e79", "-"),
        ("T2_quench_cpu_kappa", r"Staggered $\kappa(d)$, $G_c(T)$", "#2a9d8f", "-"),
        ("T2_quench_lbfgs_k0gct", r"L-BFGS $\kappa_0$, $G_c(T)$", "#c0392b", "--"),
        ("T2_quench_lbfgs_kd10", r"L-BFGS $\kappa(d)$, $G_c(T)$", "#e67e22", "--"),
        ("T2_quench_pfczm_lbfgs", r"PF-CZM $40\times 16$", "#8e44ad", ":"),
        ("T2_quench_pfczm_ref", r"PF-CZM $80\times 32$", "#6c3483", "-."),
        ("T2_quench_cmame_k0", r"Staggered $\kappa_0$, $312\times 125$", "#1a5276", "-"),
        ("T2_quench_cmame_kd", r"Staggered $\kappa(d)$, $312\times 125$", "#117a65", "-"),
    ]
    for case, lab, c, ls in series:
        r = load(case)
        ax.plot(col(r, "time"), col(r, "n_cracked"), ls, color=c, lw=1.25, label=lab)
    ax.set_xlabel(r"Time $t$ [s]")
    ax.set_ylabel(r"$n_{d>0.8}$")
    ax.legend(loc="best", fontsize=7.4)
    style_ax(ax)
    fig.savefig(OUT / "t2_ncr.pdf")
    plt.close(fig)


def fig_t2_tmax():
    fig, ax = plt.subplots(figsize=(3.45, 2.65))
    for case, lab, c, ls in (
        ("T2_quench_cpu", r"Staggered $\kappa_0$", "#1f4e79", "-"),
        ("T2_quench_cpu_kappa", r"Staggered $\kappa(d)$", "#2a9d8f", "-"),
        ("T2_quench_lbfgs_k0gct", r"L-BFGS $\kappa_0$", "#c0392b", "--"),
        ("T2_quench_lbfgs_kd10", r"L-BFGS $\kappa(d)$", "#e67e22", "--"),
        ("T2_quench_cmame_k0", r"CMAME $\kappa_0$", "#1a5276", "-"),
        ("T2_quench_cmame_kd", r"CMAME $\kappa(d)$", "#117a65", "-"),
    ):
        r = load(case)
        ax.plot(col(r, "time"), col(r, "T_max"), ls, color=c, lw=1.3, label=lab)
    ax.set_xlabel(r"Time $t$ [s]")
    ax.set_ylabel(r"$T_{\max}$ [K]")
    ax.legend()
    style_ax(ax)
    fig.savefig(OUT / "t2_tmax.pdf")
    plt.close(fig)


def fig_i1():
    path = ROOT / "outputs" / "I1_gcT_inverse_cpu" / "history.csv"
    with path.open() as f:
        rows = list(csv.DictReader(f))
    it = col(rows, "iter")
    J = col(rows, "J")
    Gc0 = col(rows, "Gc0")
    b1 = col(rows, "b1")
    fig, ax = plt.subplots(figsize=(3.45, 2.65))
    ax.plot(it, J * 1e4, "-o", color="#1f4e79", ms=4.5, lw=1.2, label=r"$\mathcal{J}\times 10^{4}$")
    ax.set_xlabel("Outer iteration")
    ax.set_ylabel(r"Misfit $\mathcal{J}\times 10^{4}$")
    ax.set_xticks(list(it))
    ax2 = ax.twinx()
    ax2.plot(it, Gc0 / 4.247e-5, "-s", color="#c0392b", ms=4, lw=1.1, label=r"$G_{c0}/G_{c0}^{\star}$")
    ax2.plot(it, b1 / 1.8, "-^", color="#2a9d8f", ms=4, lw=1.1, label=r"$b_1/b_1^{\star}$")
    ax2.set_ylabel("Relative coefficient")
    ax2.set_ylim(-0.05, 1.25)
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, loc="center right")
    style_ax(ax)
    ax.tick_params(right=False)
    fig.savefig(OUT / "i1_history.pdf")
    plt.close(fig)


def crop_whitespace(path, pad=18):
    import matplotlib.image as mpimg
    im = mpimg.imread(path)
    rgb = im[:, :, :3]
    white = (rgb > 0.985).all(axis=2)
    rows = np.where(~white.any(axis=1))[0]
    cols = np.where(~white.any(axis=0))[0]
    if len(rows) == 0 or len(cols) == 0:
        return
    r0 = max(int(rows[0]) - pad, 0)
    r1 = min(int(rows[-1]) + pad + 1, im.shape[0])
    c0 = max(int(cols[0]) - pad, 0)
    c1 = min(int(cols[-1]) + pad + 1, im.shape[1])
    fig = plt.figure(figsize=((c1 - c0) / 300.0, (r1 - r0) / 300.0), dpi=300)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.imshow(im[r0:r1, c0:c1])
    ax.axis("off")
    fig.savefig(path, dpi=300)
    plt.close(fig)


def crop_all_png():
    skip = {"graphical_abstract.png"}
    for p in sorted(OUT.glob("*.png")):
        if p.name in skip or p.name.startswith("panel_"):
            continue
        crop_whitespace(p)
        print("cropped", p.name)


def main():
    fig_m1_tension()
    fig_m1_shear()
    fig_m2()
    fig_t2_ncr()
    fig_t2_tmax()
    fig_i1()
    print("curves written to", OUT)


if __name__ == "__main__":
    main()
