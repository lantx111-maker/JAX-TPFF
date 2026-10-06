#!/usr/bin/env python3
"""Create the calculation-backed manuscript outline / graphical abstract."""

from __future__ import annotations

import os
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".mplconfig"))

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyBboxPatch
from matplotlib.ticker import NullLocator


FIG_DIR = ROOT / "paper_artifacts" / "figures"
DATA = ROOT / "results" / "verification" / "profile_transport" / "transport_convergence.csv"
OUT_PDF = ROOT / "output" / "pdf" / "JAX_TPFF_Manuscript_Outline.pdf"
OUT_PNG = ROOT / "output" / "pdf" / "JAX_TPFF_Manuscript_Outline.png"

NAVY = "#12324A"
BLUE = "#146C94"
CYAN = "#2AA6B8"
RED = "#B33B32"
ORANGE = "#E47A2E"
INK = "#24313A"
MUTED = "#5E6B73"
PALE = "#F4F7F9"
GRID = "#D7E0E5"
SLATE = "#7B8993"

mpl.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "mathtext.fontset": "stixsans",
        "font.size": 9.2,
        "axes.labelsize": 9.5,
        "axes.titlesize": 10.0,
        "xtick.labelsize": 8.4,
        "ytick.labelsize": 8.4,
        "xtick.direction": "in",
        "ytick.direction": "in",
        "xtick.top": False,
        "ytick.right": False,
        "axes.spines.top": True,
        "axes.spines.right": True,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    }
)


def _card(fig: plt.Figure, x: float, y: float, w: float, h: float, edge: str) -> None:
    fig.patches.append(
        FancyBboxPatch(
            (x, y),
            w,
            h,
            boxstyle="round,pad=0.008,rounding_size=0.015",
            transform=fig.transFigure,
            facecolor="white",
            edgecolor=edge,
            linewidth=0.82,
            zorder=-5,
        )
    )


def _field_crop(name: str) -> np.ndarray:
    image = plt.imread(FIG_DIR / name)
    return image[115:985, 413:2587]


def main() -> None:
    fig = plt.figure(figsize=(15.6, 5.2), facecolor=PALE)

    # High-contrast title band; the one-line subtitle states the complete story.
    fig.patches.append(
        FancyBboxPatch(
            (0.015, 0.875),
            0.970,
            0.105,
            boxstyle="round,pad=0.004,rounding_size=0.014",
            transform=fig.transFigure,
            facecolor=NAVY,
            edgecolor=NAVY,
            linewidth=0,
            zorder=-5,
        )
    )
    fig.text(0.5, 0.944, "Diffuse cracks encode a finite thermal resistance", ha="center", va="center", color="white", fontsize=20.2, fontweight="bold")
    fig.text(0.5, 0.898, "EXACT RESISTANCE  |  TRANSPORT RESOLUTION  |  THERMAL-SHOCK CONSEQUENCE", ha="center", va="center", color="#DCEAF0", fontsize=10.8, fontweight="bold")

    cards = [
        (0.020, 0.035, 0.270, 0.800, SLATE),
        (0.305, 0.035, 0.300, 0.800, SLATE),
        (0.620, 0.035, 0.360, 0.800, SLATE),
    ]
    for x0, y0, w0, h0, colour in cards:
        _card(fig, x0, y0, w0, h0, colour)

    headers = [
        (0.044, "1", "EXACT RESISTANCE", RED, 13.2),
        (0.329, "2", "RESISTANCE CALIBRATION · FLUX CONVERGENCE", BLUE, 10.9),
        (0.644, "3", "THERMAL-SHOCK CONSEQUENCE", CYAN, 13.0),
    ]
    for x0, number, title, colour, size in headers:
        fig.text(x0, 0.786, number, ha="center", va="center", color="white", fontsize=11.0, fontweight="bold", bbox={"boxstyle": "circle,pad=0.36", "fc": colour, "ec": colour})
        fig.text(x0 + 0.027, 0.786, title, ha="left", va="center", color=NAVY, fontsize=size, fontweight="bold")

    # Stage 1: profile, narrow resistance core and exact excess resistance.
    ax = fig.add_axes([0.048, 0.305, 0.214, 0.365], facecolor="none")
    x = np.linspace(-4.0, 4.0, 600)
    d = np.exp(-np.abs(x))
    k_res = 1.0e-2
    gk = k_res + (1.0 - k_res) * (1.0 - d) ** 2
    ax.fill_between(x, 0, d, color=RED, alpha=0.10, lw=0)
    ax.plot(x, d, color=RED, lw=2.3, label=r"damage $d$")
    ax.plot(x, gk, color=BLUE, lw=2.3, label=r"$\kappa(d)/\kappa_0$")
    ax.axvspan(-np.sqrt(k_res), np.sqrt(k_res), color=ORANGE, alpha=0.19, lw=0)
    ax.set(xlim=(-4, 4), ylim=(0, 1.04), xlabel=r"$x/\ell$", ylabel="")
    ax.legend(frameon=False, fontsize=8.8, loc="center right", handlelength=2.4)
    ax.grid(axis="y", color=GRID, lw=0.55)
    ax.annotate(
        "resistance-dominating\nregion\n" + r"$O(\ell\sqrt{k_\kappa})$",
        xy=(0.02, 0.69),
        xytext=(0.75, 0.88),
        ha="left",
        va="center",
        fontsize=8.0,
        color=MUTED,
        arrowprops={"arrowstyle": "-", "color": ORANGE, "lw": 0.9},
        bbox={"boxstyle": "round,pad=0.18", "fc": "white", "ec": "none", "alpha": 0.90},
    )
    fig.text(0.155, 0.115, r"$R_{\rm pf}=\frac{\ell}{\kappa_0}I(k_\kappa)$", ha="center", va="center", fontsize=17.5, color=NAVY, fontweight="bold", bbox={"boxstyle": "round,pad=0.30", "fc": "#EEF4F7", "ec": NAVY, "lw": 0.9})

    # Stage 2: simplify the trend plot and separate calibration from resolution.
    ax = fig.add_axes([0.337, 0.350, 0.236, 0.332], facecolor="none")
    data = np.genfromtxt(DATA, delimiter=",", names=True)
    colours = {1e-2: BLUE, 1e-3: ORANGE, 1e-4: RED}
    markers = {1e-2: "o", 1e-3: "s", 1e-4: "^"}
    for value in (1e-2, 1e-3, 1e-4):
        subset = data[np.isclose(data["k_res"], value)]
        nth = subset["ell_over_h"] * np.sqrt(value)
        error = 100.0 * subset["heat_flux_relative_error"]
        ax.loglog(nth, error, marker=markers[value], ms=4.8, color=colours[value], lw=2.0, label=rf"$10^{{{int(np.log10(value))}}}$")
    ax.axhline(5.0, color="0.42", ls="--", lw=0.85)
    ax.set(xlabel=r"thermal-transport resolution $N_{\rm th}$", ylabel="heat-flux error [%]", ylim=(8e-3, 1e3))
    ax.grid(which="major", color=GRID, lw=0.55)
    ax.legend(title=r"$k_\kappa$", frameon=False, fontsize=8.0, title_fontsize=8.2, loc="lower left", handlelength=1.6)
    ax.xaxis.set_minor_locator(NullLocator())
    ax.yaxis.set_minor_locator(NullLocator())
    fig.text(0.455, 0.255, r"phase-field resolution $\neq$ transport convergence", ha="center", va="center", color=NAVY, fontsize=10.9, fontweight="bold")
    for x0, w0 in ((0.322, 0.128), (0.460, 0.128)):
        fig.patches.append(
            FancyBboxPatch(
                (x0, 0.060),
                w0,
                0.155,
                boxstyle="round,pad=0.004,rounding_size=0.010",
                transform=fig.transFigure,
                facecolor="#F3F7F9",
                edgecolor="#C2CFD6",
                linewidth=0.75,
                zorder=-1,
            )
        )
    fig.text(0.386, 0.153, r"$I(k_\kappa)=\kappa_0R_*/\ell$", ha="center", va="center", color=NAVY, fontsize=13.5, fontweight="bold")
    fig.text(0.386, 0.092, "resistance-equivalent calibration", ha="center", va="center", color=MUTED, fontsize=8.4)
    fig.text(0.524, 0.153, r"$N_{\rm th}=\ell\sqrt{k_\kappa}/h$", ha="center", va="center", color=BLUE, fontsize=13.0, fontweight="bold")
    fig.text(0.524, 0.092, "transport-resolution criterion", ha="center", va="center", color=MUTED, fontsize=8.4)

    # Stage 3: matched fields expose the thermal-first causal sequence.
    fig.text(0.742, 0.704, r"constant $\kappa_0$", ha="center", va="center", color=INK, fontsize=10.5, fontweight="bold")
    fig.text(0.892, 0.704, r"damage-dependent $\kappa(d)$", ha="center", va="center", color=INK, fontsize=10.5, fontweight="bold")
    image_specs = [
        ("manuscript_main_k0_T.png", 0.674, 0.545),
        ("manuscript_main_kd_T.png", 0.824, 0.545),
        ("manuscript_main_k0_q.png", 0.674, 0.365),
        ("manuscript_main_kd_q.png", 0.824, 0.365),
        ("manuscript_main_k0_d.png", 0.674, 0.185),
        ("manuscript_main_kd_d.png", 0.824, 0.185),
    ]
    for name, left, bottom in image_specs:
        image_ax = fig.add_axes([left, bottom, 0.136, 0.118])
        image_ax.imshow(_field_crop(name), interpolation="none")
        image_ax.set_axis_off()
    fig.text(0.646, 0.604, r"$T$", ha="center", va="center", color=RED, fontsize=15.0, fontweight="bold")
    fig.text(0.646, 0.424, r"$|\mathbf{q}|$", ha="center", va="center", color=BLUE, fontsize=14.0, fontweight="bold")
    fig.text(0.646, 0.244, r"$d$", ha="center", va="center", color=ORANGE, fontsize=15.0, fontweight="bold")
    fig.text(0.800, 0.100, r"higher resistance $\rightarrow$ lower $|\mathbf{q}|$ $\rightarrow$ heat retention $\rightarrow$ weaker fracture drive", ha="center", va="center", color=NAVY, fontsize=9.1, fontweight="bold")

    OUT_PDF.parent.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_PDF, bbox_inches="tight", pad_inches=0.045, facecolor=fig.get_facecolor())
    fig.savefig(OUT_PNG, dpi=420, bbox_inches="tight", pad_inches=0.045, facecolor=fig.get_facecolor())
    plt.close(fig)

    for name in ("manuscript_outline.pdf", "graphical_abstract.pdf"):
        shutil.copy2(OUT_PDF, FIG_DIR / name)
    for name in ("manuscript_outline.png", "graphical_abstract.png"):
        shutil.copy2(OUT_PNG, FIG_DIR / name)


if __name__ == "__main__":
    main()
