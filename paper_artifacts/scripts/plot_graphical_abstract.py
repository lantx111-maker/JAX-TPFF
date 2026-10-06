#!/usr/bin/env python3
"""Transport-first graphical abstract for the IJHMT submission."""

from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".mplconfig"))

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.gridspec import GridSpec

FIG = ROOT / "paper_artifacts" / "figures"
DATA = ROOT / "results" / "verification" / "profile_transport" / "transport_convergence.csv"

mpl.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
    "mathtext.fontset": "stix",
    "font.size": 9.0,
    "axes.labelsize": 8.2,
    "axes.titlesize": 9.3,
    "xtick.direction": "in",
    "ytick.direction": "in",
    "xtick.top": False,
    "ytick.right": False,
    "axes.spines.top": True,
    "axes.spines.right": True,
    "pdf.fonttype": 42,
})


def field_crop(name: str):
    image = plt.imread(FIG / name)
    return image[115:985, 413:2587]


fig = plt.figure(figsize=(12.4, 3.55))
grid = GridSpec(1, 4, figure=fig, width_ratios=(1.0, 1.15, 1.15, 1.55), wspace=0.42)

# (a) Diffuse profile and the conductivity closure.
ax = fig.add_subplot(grid[0, 0])
x = np.linspace(-4.0, 4.0, 600)
d = np.exp(-np.abs(x))
kres = 1.0e-2
gk = kres + (1.0 - kres) * (1.0 - d) ** 2
ax.plot(x, d, color="#b2182b", lw=2.0, label=r"$d$")
ax.plot(x, gk, color="#2166ac", lw=2.0, label=r"$\kappa/\kappa_0$")
ax.axvspan(-np.sqrt(kres), np.sqrt(kres), color="#f4a582", alpha=0.34, lw=0)
ax.set(xlabel=r"normal coordinate $x/\ell$", ylabel="normalised field", xlim=(-4, 4), ylim=(0, 1.04))
ax.legend(loc="center right", frameon=False)
ax.set_title("(a) Diffuse conductivity layer", loc="left", fontweight="bold")

# (b) Resistance interpretation and calibration.
ax = fig.add_subplot(grid[0, 1])
ax.axis("off")
ax.set_title("(b) Resistance and calibration", loc="left", fontweight="bold")
ax.text(0.50, 0.76, r"$R_{\rm pf}=\dfrac{\ell}{\kappa_0}I(k_\kappa)$", ha="center", va="center", fontsize=15)
ax.annotate("", xy=(0.50, 0.57), xytext=(0.50, 0.67), arrowprops={"arrowstyle": "-|>", "lw": 1.3, "color": "#444444"})
ax.text(0.50, 0.49, r"$I(k_\kappa(\ell))=\kappa_0R^*/\ell$", ha="center", va="center", fontsize=12.5)
ax.text(0.50, 0.28, "Prescribed resistance retained\nwhen the phase-field length changes", ha="center", va="center", linespacing=1.35)
ax.text(0.50, 0.09, r"$R_{\rm pf}=R_{\rm int}=1/h_{\rm int}$ for 1D normal mapping", ha="center", va="center", fontsize=8.3, color="#444444")

# (c) Transport-resolution scaling.
ax = fig.add_subplot(grid[0, 2])
data = np.genfromtxt(DATA, delimiter=",", names=True)
colours = {1e-2: "#2166ac", 1e-3: "#ef8a62", 1e-4: "#b2182b"}
markers = {1e-2: "o", 1e-3: "s", 1e-4: "^"}
for value in (1e-2, 1e-3, 1e-4):
    subset = data[np.isclose(data["k_res"], value)]
    nth = subset["ell_over_h"] * np.sqrt(value)
    error = 100.0 * subset["heat_flux_relative_error"]
    ax.loglog(nth, error, marker=markers[value], ms=4, color=colours[value], label=rf"$10^{{{int(np.log10(value))}}}$")
ax.axhline(5.0, color="0.45", ls="--", lw=0.8)
ax.axhline(1.0, color="0.45", ls=":", lw=0.8)
ax.axvspan(0.64, 0.91, color="#fddbc7", alpha=0.38, lw=0)
ax.axvspan(1.36, 1.75, color="#d1e5f0", alpha=0.38, lw=0)
ax.set(xlabel=r"$N_{\rm th}=\ell\sqrt{k_\kappa}/h$", ylabel="heat-flux error [%]", ylim=(8e-3, 1e3))
ax.legend(title=r"$k_\kappa$", ncol=1, frameon=False, fontsize=7, title_fontsize=7)
ax.set_title("(c) Transport resolution", loc="left", fontweight="bold")

# (d) System-level consequence in the ceramic quench.
cell = grid[0, 3].subgridspec(3, 2, height_ratios=(0.75, 0.75, 0.38), hspace=0.22, wspace=0.08)
left = fig.add_subplot(cell[0, 0])
right = fig.add_subplot(cell[0, 1])
left.imshow(field_crop("manuscript_main_k0_T.png"), interpolation="none")
right.imshow(field_crop("manuscript_main_kd_T.png"), interpolation="none")
left.set_title(r"$\kappa_0$", fontsize=8.5)
right.set_title(r"$\kappa(d)$", fontsize=8.5)
damage_left = fig.add_subplot(cell[1, 0])
damage_right = fig.add_subplot(cell[1, 1])
damage_left.imshow(field_crop("manuscript_main_k0_d.png"), interpolation="none")
damage_right.imshow(field_crop("manuscript_main_kd_d.png"), interpolation="none")
for item in (left, right, damage_left, damage_right):
    item.set_axis_off()
title_ax = fig.add_subplot(cell[:, :])
title_ax.axis("off")
title_ax.set_title("(d) Ceramic-quench consequence", loc="left", fontweight="bold", pad=1)
chain = fig.add_subplot(cell[2, :])
chain.axis("off")
chain.text(0.50, 0.54, r"$R_{\rm layer}\!\uparrow\;\Rightarrow\;Q_{\rm out}\!\downarrow\;70.2\%\;\Rightarrow\;\overline{T}_{\rm core}\!\uparrow\;42.11\,\mathrm{K}\;\Rightarrow\;\Gamma_\ell\!\downarrow$", ha="center", va="center", fontsize=9.2, fontweight="bold")

fig.suptitle("Diffuse cracks introduce a finite, resolution-sensitive thermal resistance", fontsize=13.0, fontweight="bold", y=0.995)
fig.subplots_adjust(left=0.045, right=0.995, bottom=0.16, top=0.86)
fig.savefig(FIG / "graphical_abstract.pdf", bbox_inches="tight", pad_inches=0.03)
fig.savefig(FIG / "graphical_abstract.png", dpi=600, bbox_inches="tight", pad_inches=0.03)
plt.close(fig)
