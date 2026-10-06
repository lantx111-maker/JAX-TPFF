#!/usr/bin/env python3
"""Compose cropped ParaView fields into labelled journal panels."""

from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from mpl_toolkits.axes_grid1 import ImageGrid

FIG = Path(__file__).resolve().parents[1] / "figures"

mpl.rcParams.update(
    {
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
        "mathtext.fontset": "stix",
        "font.size": 9,
        "axes.titlesize": 9,
        "pdf.fonttype": 42,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.02,
    }
)

D_CMAP = LinearSegmentedColormap.from_list(
    "dfield",
    [
        (0.00, (0.97, 0.97, 0.96)),
        (0.20, (0.99, 0.90, 0.55)),
        (0.50, (0.96, 0.50, 0.12)),
        (0.80, (0.78, 0.08, 0.12)),
        (1.00, (0.28, 0.00, 0.08)),
    ],
)
T_CMAP = LinearSegmentedColormap.from_list(
    "tfield",
    [
        (0.00, (0.12, 0.28, 0.72)),
        (0.31, (0.25, 0.72, 0.78)),
        (0.67, (0.98, 0.88, 0.35)),
        (1.00, (0.86, 0.18, 0.10)),
    ],
)


def load_rgb(path: Path) -> np.ndarray:
    im = plt.imread(path)
    if im.dtype.kind == "u":
        im = im.astype(np.float64) / 255.0
    else:
        im = im.astype(np.float64)
    if im.ndim == 2:
        im = np.repeat(im[:, :, None], 3, axis=2)
    return np.clip(im[:, :, :3], 0.0, 1.0)


def extract_plate(path: Path, drop_right: float = 0.155, pad: int = 2) -> np.ndarray:
    rgb = load_rgb(path)
    h, w = rgb.shape[:2]
    rgb = rgb[:, : max(int(w * (1.0 - drop_right)), w // 2)]
    white = (rgb > 0.992).all(axis=2)
    rows = np.where(~white.all(axis=1))[0]
    cols = np.where(~white.all(axis=0))[0]
    if len(rows) == 0 or len(cols) == 0:
        return rgb
    r0 = max(int(rows[0]) - pad, 0)
    r1 = min(int(rows[-1]) + pad + 1, rgb.shape[0])
    c0 = max(int(cols[0]) - pad, 0)
    c1 = min(int(cols[-1]) + pad + 1, rgb.shape[1])
    return rgb[r0:r1, c0:c1]


def _letter(ax, text: str) -> None:
    ax.text(
        0.015,
        0.97,
        text,
        transform=ax.transAxes,
        va="top",
        ha="left",
        fontsize=10,
        fontweight="bold",
        color="0.08",
        bbox=dict(boxstyle="square,pad=0.12", fc="white", ec="none", alpha=0.72),
    )


def grid_d(files, labels, out, nrows, ncols, figsize, cbar_label=r"$d$", drop=0.155):
    images = [extract_plate(FIG / f, drop_right=drop) for f in files]
    fig = plt.figure(figsize=figsize)
    grid = ImageGrid(
        fig,
        111,
        nrows_ncols=(nrows, ncols),
        axes_pad=(0.18, 0.32),
        cbar_mode="single",
        cbar_location="right",
        cbar_size="3.2%",
        cbar_pad=0.08,
    )
    last = None
    axes = list(grid)
    for ax, im, lab, let in zip(axes, images, labels, "abcdefghijklmnopqrstuvwxyz"):
        last = ax.imshow(im, aspect="equal")
        ax.set_axis_off()
        ax.set_title(lab, pad=3)
        _letter(ax, f"({let})")
    for ax in axes[len(images):]:
        ax.set_axis_off()
    sm = mpl.cm.ScalarMappable(cmap=D_CMAP, norm=mpl.colors.Normalize(0, 1))
    sm.set_array([])
    cb = fig.colorbar(sm, cax=grid.cbar_axes[0])
    cb.set_label(cbar_label)
    cb.ax.tick_params(labelsize=8)
    fig.savefig(FIG / out)
    plt.close(fig)
    print("wrote", out)
    return last


def grid_mixed(pairs, out, figsize, cbar="T", ncols=2):
    """pairs: list of (file, label, kind) with kind in {'d','T'}."""
    n = len(pairs)
    nrows = int(np.ceil(n / ncols))
    fig = plt.figure(figsize=figsize)
    grid = ImageGrid(
        fig,
        111,
        nrows_ncols=(nrows, ncols),
        axes_pad=(0.18, 0.32),
        cbar_mode="single" if cbar else None,
        cbar_location="right",
        cbar_size="3.2%",
        cbar_pad=0.08,
    )
    axes = list(grid)
    for ax, (fname, lab, kind), let in zip(axes, pairs, "abcdefghijklmnopqrstuvwxyz"):
        drop = 0.12 if "t3" in fname else 0.155
        im = extract_plate(FIG / fname, drop_right=drop)
        ax.imshow(im, aspect="equal")
        ax.set_axis_off()
        ax.set_title(lab, pad=3)
        _letter(ax, f"({let})")
    for ax in axes[len(pairs):]:
        ax.set_axis_off()
    if cbar:
        if cbar == "T":
            sm = mpl.cm.ScalarMappable(cmap=T_CMAP, norm=mpl.colors.Normalize(293.15, 573.15))
            label = r"$T$ [K]"
        else:
            sm = mpl.cm.ScalarMappable(cmap=D_CMAP, norm=mpl.colors.Normalize(0, 1))
            label = r"$d$"
        sm.set_array([])
        cb = fig.colorbar(sm, cax=grid.cbar_axes[0])
        cb.set_label(label)
        cb.ax.tick_params(labelsize=8)
    fig.savefig(FIG / out)
    plt.close(fig)
    print("wrote", out)


def main():
    grid_d(
        [
            "m1_tension_ref_d.png",
            "m1_tension_lbfgs_ref_d.png",
            "m1_tension_amr_d.png",
            "m1_tension_at1_d.png",
            "m1_shear_ref_ext_d.png",
            "m1_shear_lbfgs_ref_d.png",
        ],
        [
            r"AT2 staggered $80\times 80$",
            r"AT2 L-BFGS $80\times 80$",
            r"AT2 AMR TRI3",
            r"AT1 staggered $40\times 40$",
            r"AT2 shear $80\times 80$, $u=0.020$",
            r"AT2 shear L-BFGS $80\times 80$",
        ],
        "panel_m1_fields.pdf",
        nrows=2,
        ncols=3,
        figsize=(7.2, 5.15),
    )

    grid_d(
        ["m1_shear_lbfgs_d.png", "m1_shear_amr_d.png", "m2_unload_d.png", "m2_reload_d.png"],
        [
            r"Shear L-BFGS $40\times 40$",
            r"Shear AMR TRI3",
            r"M2 unload ($u=0.002\,\mathrm{mm}$)",
            r"M2 reload ($u=0.012\,\mathrm{mm}$)",
        ],
        "panel_m1_shear_m2.pdf",
        nrows=2,
        ncols=2,
        figsize=(5.9, 5.55),
    )

    grid_mixed(
        [
            ("t1_ref_d.png", r"T1 $80\times 80$ damage $d$", "d"),
            ("t1_ref_T.png", r"T1 $80\times 80$ temperature $T$", "T"),
        ],
        "panel_t1.pdf",
        figsize=(6.4, 3.15),
        cbar=None,
    )

    grid_d(
        ["t2_k0_d.png", "t2_kd_d.png", "t2_lbfgs_k0gct_d.png", "t2_lbfgs_kd10_d.png"],
        [
            r"Staggered $\kappa_0$, $G_c(T)$",
            r"Staggered $\kappa(d)$, $G_c(T)$",
            r"Block LMQN $\kappa_0$, $G_c(T)$",
            r"Block LMQN $\kappa(d)$, $G_c(T)$",
        ],
        "panel_t2_damage.pdf",
        nrows=2,
        ncols=2,
        figsize=(7.15, 3.55),
    )

    grid_mixed(
        [
            ("t2_k0_T.png", r"Staggered $\kappa_0$, $T$", "T"),
            ("t2_kd_T.png", r"Staggered $\kappa(d)$, $T$", "T"),
            ("t2_ref_T.png", r"Production $\kappa(d)$, $T$", "T"),
            ("t2_reheat_T.png", r"L-BFGS-B reheat, $T$", "T"),
        ],
        "panel_t2_temp.pdf",
        figsize=(7.15, 3.55),
    )

    grid_d(
        ["t2_ref_d.png", "t2_pfczm_d.png", "t2_pfczm_ref_d.png", "t2_amr_d.png"],
        [
            r"Production AT2 $\kappa(d)$, $125\times 50$",
            r"PF-CZM $40\times 16$ (flooded)",
            r"PF-CZM $80\times 32$ (localised)",
            r"AMR TRI3 $\kappa(d)$",
        ],
        "panel_t2_models.pdf",
        nrows=2,
        ncols=2,
        figsize=(7.15, 3.55),
    )

    grid_d(
        [
            "t3_mid_d.png",
            "t3_cmame_d.png",
        ],
        [
            r"T3 HEX8 $16\times 8\times 5$",
            r"T3 HEX8 $40\times 16\times 10$",
        ],
        "panel_t3.pdf",
        nrows=1,
        ncols=2,
        figsize=(7.15, 3.2),
        drop=0.09,
    )

    grid_d(
        [
            "m1_tension_cmame_d.png",
            "m1_shear_cmame_d.png",
            "t2_cmame_k0_d.png",
            "t2_cmame_kd_d.png",
        ],
        [
            r"M1 tension $256\times 256$, $\ell/h=3.84$",
            r"M1 shear $256\times 256$, $u=0.020$",
            r"T2 $\kappa_0$, $312\times 125$, $\ell/h=5$",
            r"T2 $\kappa(d)$, $312\times 125$, $\ell/h=5$",
        ],
        "panel_cmame.pdf",
        nrows=2,
        ncols=2,
        figsize=(7.15, 5.4),
    )

    grid_d(
        [
            "m1_tension_amr_cmame_d.png",
            "t2_amr_cmame_d.png",
        ],
        [
            r"M1 tension AMR, $50\,770$ TRI3, $\ell/h_{\min}=27$",
            r"T2 $\kappa(d)$ AMR, $51\,809$ TRI3, $\ell/h_{\min}=18$",
        ],
        "panel_cmame_amr.pdf",
        nrows=1,
        ncols=2,
        figsize=(7.15, 3.15),
    )

    grid_d(
        [
            "m3_tpb_d.png",
            "m5_hole_d.png",
            "m1_shear_pfczm_d.png",
            "m4_lshape_d.png",
        ],
        [
            r"M3 three-point bending, $|u|=0.04$",
            r"M5 notched plate with a hole, $u=0.012$",
            r"PF-CZM shear, $80\times 80$, $u=0.020$",
            r"M4 L-panel, $u=0.10$",
        ],
        "panel_benchmarks.pdf",
        nrows=2,
        ncols=2,
        figsize=(7.15, 5.6),
    )

    grid_mixed(
        [
            ("t2_cmame_k0_T.png", r"T2 $\kappa_0$, $312\times 125$", "T"),
            ("t2_cmame_kd_T.png", r"T2 $\kappa(d)$, $312\times 125$", "T"),
            ("t2_amr_cmame_T.png", r"T2 $\kappa(d)$ AMR, $\ell/h_{\min}=18$", "T"),
        ],
        "panel_cmame_temp.pdf",
        figsize=(7.15, 2.55),
        ncols=3,
    )

    # Five fields: three quench plates on the top row, two squares centred below.
    lim_files = [
        "t2_pfczm_cmame_d.png",
        "t2_lbfgs_cmame_k0_d.png",
        "t2_lbfgs_cmame_kd_d.png",
        "m1_tension_lbfgs_cmame_d.png",
        "m1_shear_lbfgs_cmame_d.png",
    ]
    lim_labs = [
        r"PF-CZM staggered, $312\times 125$",
        r"L-BFGS $\kappa_0$, $312\times 125$",
        r"L-BFGS $\kappa(d)$, $312\times 125$",
        r"L-BFGS tension, $256\times 256$",
        r"L-BFGS shear, $256\times 256$",
    ]
    lim_images = [extract_plate(FIG / f) for f in lim_files]
    fig = plt.figure(figsize=(7.15, 4.7))
    gs = fig.add_gridspec(2, 6, hspace=0.32, wspace=0.35)
    slots = [(0, slice(0, 2)), (0, slice(2, 4)), (0, slice(4, 6)), (1, slice(1, 3)), (1, slice(3, 5))]
    for (im, lab, let, (row, cols)) in zip(lim_images, lim_labs, "abcde", slots):
        ax = fig.add_subplot(gs[row, cols])
        ax.imshow(im, aspect="equal")
        ax.set_axis_off()
        ax.set_title(lab, pad=3)
        _letter(ax, f"({let})")
    fig.subplots_adjust(right=0.90)
    cax = fig.add_axes([0.915, 0.18, 0.018, 0.64])
    sm = mpl.cm.ScalarMappable(cmap=D_CMAP, norm=mpl.colors.Normalize(0, 1))
    sm.set_array([])
    cb = fig.colorbar(sm, cax=cax)
    cb.set_label(r"$d$")
    cb.ax.tick_params(labelsize=8)
    fig.savefig(FIG / "panel_cmame_limits.pdf")
    plt.close(fig)
    print("wrote panel_cmame_limits.pdf")

    grid_mixed(
        [
            ("t3_mid_T.png", r"T3 $16\times 8\times 5$, $T$", "T"),
            ("t3_cmame_T.png", r"T3 $40\times 16\times 10$, $T$", "T"),
        ],
        "panel_t3_temp.pdf",
        figsize=(7.15, 3.2),
    )

    grid_d(
        ["i1_cpu_obs_d.png", "i1_cpu_hat_d.png"],
        [
            r"Hidden $G_c(T)$ on $40\times 16$",
            r"Recovered $(b_1,b_2)\approx(0.07,0.007)$",
        ],
        "panel_i1.pdf",
        nrows=1,
        ncols=2,
        figsize=(7.15, 2.05),
    )

    # Graphical abstract: conductivity split + PF-CZM over-damage.
    ga_files = ["t2_k0_d.png", "t2_kd_d.png", "t2_pfczm_ref_d.png"]
    ga_labs = [
        r"$\kappa=\kappa_0$: interior fingers",
        r"$\kappa(d)$: shielded skin",
        r"PF-CZM $80\times 32$: localised",
    ]
    images = [extract_plate(FIG / f) for f in ga_files]
    fig = plt.figure(figsize=(7.2, 2.35))
    grid = ImageGrid(
        fig,
        111,
        nrows_ncols=(1, 3),
        axes_pad=0.22,
        cbar_mode="single",
        cbar_location="right",
        cbar_size="4%",
        cbar_pad=0.08,
    )
    for ax, im, lab, let in zip(grid, images, ga_labs, "abc"):
        ax.imshow(im, aspect="equal")
        ax.set_axis_off()
        ax.set_title(lab, pad=3, fontsize=8.5)
        _letter(ax, f"({let})")
    sm = mpl.cm.ScalarMappable(cmap=D_CMAP, norm=mpl.colors.Normalize(0, 1))
    sm.set_array([])
    cb = fig.colorbar(sm, cax=grid.cbar_axes[0])
    cb.set_label(r"$d$")
    fig.savefig(FIG / "graphical_abstract.pdf")
    fig.savefig(FIG / "graphical_abstract.png", dpi=300)
    plt.close(fig)
    print("wrote graphical_abstract")


if __name__ == "__main__":
    main()
