#!/usr/bin/env python3
"""Generate the publication figures, tables, and manuscript macros."""

from __future__ import annotations

import csv
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".mplconfig"))

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.gridspec import GridSpec
from matplotlib.patches import Rectangle
import numpy as np


OUT = ROOT / "results" / "manuscript"
POST = ROOT / "results" / "postprocessed"
FIG = ROOT / "paper_artifacts" / "figures"
TABLES = ROOT / "paper_artifacts" / "tables"
ELL = 0.5
LX = 12.5
LY = 5.0
KAPPA0 = 0.031
KRES = 0.01
TREF = 573.15

DAMAGE_CMAP = LinearSegmentedColormap.from_list(
    "damage",
    [
        (0.97, 0.97, 0.96),
        (0.99, 0.90, 0.55),
        (0.96, 0.50, 0.12),
        (0.78, 0.08, 0.12),
        (0.28, 0.00, 0.08),
    ],
)
TEMPERATURE_CMAP = LinearSegmentedColormap.from_list(
    "temperature",
    [
        (0.267004, 0.004874, 0.329415),
        (0.229739, 0.322361, 0.545706),
        (0.127568, 0.566949, 0.550556),
        (0.369214, 0.788888, 0.382914),
        (0.993248, 0.906157, 0.143936),
    ],
)

mpl.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
    "mathtext.fontset": "stix",
    "font.size": 8.5,
    "axes.labelsize": 8.5,
    "axes.titlesize": 8.5,
    "legend.fontsize": 7.0,
    "xtick.direction": "in",
    "ytick.direction": "in",
    "xtick.top": False,
    "ytick.right": False,
    "axes.spines.top": True,
    "axes.spines.right": True,
    "axes.grid": True,
    "grid.alpha": 0.17,
    "grid.linewidth": 0.4,
    "lines.linewidth": 1.25,
    "pdf.fonttype": 42,
    "image.composite_image": False,
})


def _value(text: str):
    if text == "True":
        return True
    if text == "False":
        return False
    return float(text)


def read_history(name: str):
    with (OUT / name / "history.csv").open() as stream:
        return [{key: _value(value) for key, value in row.items()} for row in csv.DictReader(stream)]


def state_history(name: str, key: str, initial: float):
    rows = read_history(name)
    return np.array([0.0] + [row["time"] for row in rows]), np.array([initial] + [row[key] for row in rows])


def rate_history(name: str, key: str):
    rows = read_history(name)
    return np.array([row["time"] for row in rows]), np.array([row[key] for row in rows])


def style_axes(axes):
    for ax in np.ravel(np.atleast_1d(axes)):
        ax.minorticks_on()
        ax.tick_params(which="both", direction="in", top=False, right=False)
        for spine in ax.spines.values():
            spine.set_visible(True)


def panel_labels(axes, *, image_panels=False, label_y=None):
    """Place panel identifiers below each panel without touching axis labels."""
    for index, ax in enumerate(np.ravel(np.atleast_1d(axes))):
        has_xlabel = bool(ax.get_xlabel())
        if has_xlabel and not image_panels:
            # Keep the axis title close to the tick labels. The panel tag is
            # placed on a separate line below it: ticks -> title -> tag.
            ax.xaxis.set_label_coords(0.5, -0.14)
        tag_y = (
            label_y
            if label_y is not None
            else (-0.115 if image_panels else (-0.32 if has_xlabel else -0.16))
        )
        ax.text(
            0.5,
            tag_y,
            f"({chr(ord('a') + index)})",
            transform=ax.transAxes,
            ha="center",
            va="top",
            fontweight="bold",
            clip_on=False,
        )


def save(fig, name: str, *, rect=None, w_pad=0.8, h_pad=0.8):
    style_axes(fig.axes)
    fig.tight_layout(rect=rect, w_pad=w_pad, h_pad=h_pad)
    fig.savefig(FIG / name, bbox_inches="tight", pad_inches=0.03)
    plt.close(fig)


def primary_names():
    return [
        f"TS_{nx:03d}x{ny:03d}_n{steps:03d}_{law}"
        for nx, ny, steps in ((100, 40, 40), (100, 40, 80), (200, 80, 40), (200, 80, 80), (400, 160, 40))
        for law in ("k0", "kd")
    ]


def collect(names):
    rows = []
    for name in names:
        hist = read_history(name)
        parts = name.split("_")
        mesh = next((part for part in parts if "x" in part and part[0].isdigit()), "100x040")
        nsteps = int(next(part[1:] for part in parts if part.startswith("n") and part[1:].isdigit()))
        if len(hist) != nsteps:
            raise RuntimeError(f"Incomplete history for {name}: {len(hist)}/{nsteps}")
        final = hist[-1]
        rows.append({
            "case": name,
            "mesh": mesh.lstrip("0").replace("x0", "x"),
            "steps": nsteps,
            "law": parts[-1],
            "T_core_K": final["T_core_mean"],
            "Gamma_mm": final["fracture_surface"],
            "Q_out_J": final["heat_out_cumulative"],
            "R_layer": final.get("R_layer_mean", 0.0),
            "max_heat_balance_rel": max(abs(row["heat_balance_relative"]) for row in hist),
            "max_kkt": max(abs(row["damage_kkt_residual"]) for row in hist),
            "all_steps_converged": all(row["converged"] and row["damage_active_set_converged"] for row in hist),
        })
    return rows


def _grid_from_checkpoint(name: str):
    parts = name.split("_")
    mesh = next(part for part in parts if "x" in part and part[0].isdigit())
    nx, ny = (int(value) for value in mesh.split("x"))
    data = np.load(OUT / name / "checkpoint.npz", allow_pickle=True)
    damage = np.asarray(data["sol_d"]).reshape((nx + 1, ny + 1)).T
    xs = np.linspace(0.0, LX, nx + 1)
    ys = np.linspace(0.0, LY, ny + 1)
    return xs, ys, damage


def _line_resistance(coord, damage):
    coord = np.asarray(coord)
    damage = np.asarray(damage)
    value = 0.0
    for index in range(len(coord) - 1):
        h = coord[index + 1] - coord[index]
        for xi in (-1.0 / np.sqrt(3.0), 1.0 / np.sqrt(3.0)):
            n0, n1 = 0.5 * (1.0 - xi), 0.5 * (1.0 + xi)
            dq = n0 * damage[index] + n1 * damage[index + 1]
            gk = KRES + (1.0 - KRES) * (1.0 - dq) ** 2
            value += (1.0 / (KAPPA0 * gk) - 1.0 / KAPPA0) * h / 2.0
    return float(value)


def boundary_paths(name: str, depth_multiple: float):
    xs, ys, damage = _grid_from_checkpoint(name)
    depth_x = min(depth_multiple * ELL, LX)
    depth_y = min(depth_multiple * ELL, 0.5 * LY)
    use_x = xs <= depth_x + 1.0e-12
    use_bottom = ys <= depth_y + 1.0e-12
    use_top = ys >= LY - depth_y - 1.0e-12
    paths = {"left": [], "bottom": [], "top": []}
    profiles = []
    for j, y in enumerate(ys):
        if 2.0 * ELL <= y <= LY - 2.0 * ELL:
            coord = xs[use_x]
            values = damage[j, use_x]
            paths["left"].append(_line_resistance(coord, values))
            profiles.append((coord / ELL, values))
    for i, x in enumerate(xs):
        if 2.0 * ELL <= x <= LX - ELL:
            cb = ys[use_bottom]
            db = damage[use_bottom, i]
            ct = LY - ys[use_top][::-1]
            dt = damage[use_top, i][::-1]
            paths["bottom"].append(_line_resistance(cb, db))
            paths["top"].append(_line_resistance(ct, dt))
            profiles.extend(((cb / ELL, db), (ct / ELL, dt)))
    return paths, profiles


def resistance_summary(name: str):
    payload = {"case": name, "depths": {}}
    for multiple in (3, 4, 5):
        paths, _ = boundary_paths(name, multiple)
        all_values = np.concatenate([np.asarray(paths[edge]) for edge in ("left", "bottom", "top")])
        payload["depths"][str(multiple)] = {
            "n_paths": int(len(all_values)),
            "mean": float(np.mean(all_values)),
            "median": float(np.median(all_values)),
            "p05": float(np.percentile(all_values, 5)),
            "p95": float(np.percentile(all_values, 95)),
            "min": float(np.min(all_values)),
            "max": float(np.max(all_values)),
            "edges": {
                edge: {
                    "n": int(len(values)),
                    "mean": float(np.mean(values)),
                    "median": float(np.median(values)),
                    "p05": float(np.percentile(values, 5)),
                    "p95": float(np.percentile(values, 95)),
                }
                for edge, values in paths.items()
            },
        }
    return payload


def write_tables_and_macros(primary, controls, resistance):
    by_key = {(row["mesh"], row["steps"], row["law"]): row for row in primary}
    main_c = by_key[("200x80", 80, "k0")]
    main_d = by_key[("200x80", 80, "kd")]
    stats5 = resistance["depths"]["5"]

    lines = [
        r"\begin{table*}[htbp]", r"\centering",
        r"\caption{Terminal mesh--time results. $\overline R_{\rm layer}$ is the mean normal-path excess resistance integrated over $5\ell$.}",
        r"\label{tab:primary-results}", r"\small",
        r"\begin{tabular}{@{}llrrrrr@{}}", r"\toprule",
        "Mesh & $N_t$ & law & $\\bar T_{\\rm core}$ [K] & $\\Gamma_\\ell$ [mm] & $Q_{\\rm out}$ [J] & $\\overline R_{\\rm layer}$ [mm$^2$K/W] " + r"\\",
        r"\midrule",
    ]
    for index, row in enumerate(primary):
        law = r"$\kappa_0$" if row["law"] == "k0" else r"$\kappa(d)$"
        mesh = "$" + row["mesh"].replace("x", r"\times") + "$"
        resistance_value = "0.00" if row["law"] == "k0" else f"{row['R_layer']:.2f}"
        lines.append(
            f"{mesh} & {row['steps']} & {law} & {row['T_core_K']:.2f} & {row['Gamma_mm']:.3f} & "
            f"{row['Q_out_J']:.3f} & {resistance_value} \\\\"
        )
        if row["law"] == "kd" and index != len(primary) - 1:
            lines.append(r"\addlinespace[1.5pt]")
    lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table*}"])
    (TABLES / "tables_primary.tex").write_text("\n".join(lines) + "\n")

    labels = {"Bi10": r"$\mathrm{Bi}_0=10$", "free": "minimal restraint", "gcT": r"synthetic $G_c(T)$", "AT1": "AT1"}
    control_map = {}
    for row in controls:
        kind = next(key for key in labels if f"_{key}_" in row["case"])
        control_map[(kind, row["law"])] = row
    lines = [
        r"\begin{table*}[htbp]", r"\centering",
        r"\caption{Robustness controls on the $100\times40$, 80-step discretisation. These cases test the sign, rather than the converged magnitude, of the conductivity effect.}",
        r"\label{tab:control-results}", r"\small",
        r"\begin{tabular}{@{}llrrr@{}}", r"\toprule",
        "Control & law & $\\bar T_{\\rm core}$ [K] & $\\Gamma_\\ell$ [mm] & $Q_{\\rm out}$ [J] " + r"\\", r"\midrule",
    ]
    for kind in ("Bi10", "free", "gcT", "AT1"):
        for law_key in ("k0", "kd"):
            row = control_map[(kind, law_key)]
            law = r"$\kappa_0$" if law_key == "k0" else r"$\kappa(d)$"
            lines.append(f"{labels[kind]} & {law} & {row['T_core_K']:.2f} & {row['Gamma_mm']:.3f} & {row['Q_out_J']:.3f} " + r"\\")
        if kind != "AT1":
            lines.append(r"\addlinespace[2pt]")
    lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table*}"])
    (TABLES / "tables_controls.tex").write_text("\n".join(lines) + "\n")

    def reduction(a, b, key):
        return 100.0 * (a[key] - b[key]) / a[key]

    def change(a, b, key):
        return 100.0 * abs(b[key] - a[key]) / abs(a[key])

    p200c40, p200d40 = by_key[("200x80", 40, "k0")], by_key[("200x80", 40, "kd")]
    p200c80, p200d80 = by_key[("200x80", 80, "k0")], by_key[("200x80", 80, "kd")]
    p400c, p400d = by_key[("400x160", 40, "k0")], by_key[("400x160", 40, "kd")]
    macros = {
        "MainTConstant": f"{main_c['T_core_K']:.2f}",
        "MainTDamage": f"{main_d['T_core_K']:.2f}",
        "MainDeltaT": f"{main_d['T_core_K'] - main_c['T_core_K']:.2f}",
        "MainGammaConstant": f"{main_c['Gamma_mm']:.3f}",
        "MainGammaDamage": f"{main_d['Gamma_mm']:.3f}",
        "MainGammaReduction": f"{reduction(main_c, main_d, 'Gamma_mm'):.1f}",
        "MainQConstant": f"{main_c['Q_out_J']:.3f}",
        "MainQDamage": f"{main_d['Q_out_J']:.3f}",
        "MainQReduction": f"{reduction(main_c, main_d, 'Q_out_J'):.1f}",
        "MainRLayer": f"{stats5['mean']:.2f}",
        "MainRMedian": f"{stats5['median']:.2f}",
        "MainRPZeroFive": f"{stats5['p05']:.2f}",
        "MainRPNinetyFive": f"{stats5['p95']:.2f}",
        "MainRDepthThree": f"{resistance['depths']['3']['mean']:.2f}",
        "MainRDepthFour": f"{resistance['depths']['4']['mean']:.2f}",
        "MainRDepthFive": f"{stats5['mean']:.2f}",
        "MainPiRLayer": f"{KAPPA0 * stats5['mean'] / (0.5 * LY):.3f}",
        "MeshGammaConstant": f"{change(p200c40, p400c, 'Gamma_mm'):.2f}",
        "MeshGammaDamage": f"{change(p200d40, p400d, 'Gamma_mm'):.2f}",
        "MeshQConstant": f"{change(p200c40, p400c, 'Q_out_J'):.2f}",
        "MeshQDamage": f"{change(p200d40, p400d, 'Q_out_J'):.2f}",
        "MeshRDamage": f"{change(p200d40, p400d, 'R_layer'):.2f}",
        "TimeGammaConstant": f"{change(p200c40, p200c80, 'Gamma_mm'):.2f}",
        "TimeGammaDamage": f"{change(p200d40, p200d80, 'Gamma_mm'):.2f}",
        "TimeQConstant": f"{change(p200c40, p200c80, 'Q_out_J'):.2f}",
        "TimeQDamage": f"{change(p200d40, p200d80, 'Q_out_J'):.2f}",
        "MaxKKT": (
            lambda parts: rf"{parts[0]}\times10^{{{int(parts[1])}}}"
        )(f"{max(row['max_kkt'] for row in primary + controls):.2e}".split("e")),
        "MaxHeatBalance": f"{100.0 * max(row['max_heat_balance_rel'] for row in primary + controls):.3f}",
        "PlaneRotationError": "8.89\\times10^{-9}",
    }
    for kind, prefix in (("Bi10", "Bi"), ("free", "Free"), ("gcT", "GcT"), ("AT1", "ATOne")):
        a, b = control_map[(kind, "k0")], control_map[(kind, "kd")]
        macros.update({
            f"{prefix}TConstant": f"{a['T_core_K']:.2f}",
            f"{prefix}TDamage": f"{b['T_core_K']:.2f}",
            f"{prefix}GammaConstant": f"{a['Gamma_mm']:.3f}",
            f"{prefix}GammaDamage": f"{b['Gamma_mm']:.3f}",
            f"{prefix}GammaReduction": f"{reduction(a, b, 'Gamma_mm'):.1f}",
            f"{prefix}QConstant": f"{a['Q_out_J']:.3f}",
            f"{prefix}QDamage": f"{b['Q_out_J']:.3f}",
        })
    (TABLES / "results_macros.tex").write_text(
        "\n".join(f"\\newcommand{{\\{key}}}{{{value}}}" for key, value in macros.items()) + "\n"
    )


def plot_mechanism():
    fig, axes = plt.subplots(2, 3, figsize=(7.15, 4.75))
    cases = (
        ("TS_200x080_n080_k0", r"$\kappa_0$", "#2166ac", "-"),
        ("TS_200x080_n080_kd", r"$\kappa(d)$", "#b2182b", "--"),
    )
    definitions = (
        ("heat_outflow", None, r"$\dot Q_{\rm out}$ [W]", False),
        ("heat_out_cumulative", 0.0, r"$Q_{\rm out}$ [J]", True),
        ("T_core_mean", TREF, r"$\overline{T}_{\rm core}$ [K]", True),
        ("R_layer_mean", 0.0, r"$\overline{R}_{\rm layer}$ [mm$^2$K/W]", True),
        ("tensile_energy", 0.0, r"$\Psi^+$ [J]", True),
        ("fracture_surface", 0.0, r"$\Gamma_\ell$ [mm]", True),
    )
    for name, label, colour, line_style in cases:
        for ax, (key, initial, ylabel, is_state) in zip(axes.ravel(), definitions):
            t, value = state_history(name, key, initial) if is_state else rate_history(name, key)
            ax.plot(t, value, color=colour, ls=line_style, label=label)
            ax.set(xlabel=r"$t$ [s]", ylabel=ylabel, xlim=(0.0, 0.1))
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, ncol=2, loc="upper center", bbox_to_anchor=(0.5, 1.0))
    panel_labels(axes)
    save(fig, "manuscript_mechanism.pdf", rect=(0.0, 0.025, 1.0, 0.965), h_pad=3.4)


def plot_controls():
    pairs = (
        ("TS_Bi10_n080", r"$\mathrm{Bi}_0=10$"),
        ("TS_free_n080", "minimal restraint"),
        ("TS_gcT_n080", r"synthetic $G_c(T)$"),
        ("TS_AT1_n080", "AT1"),
    )
    fig, axes = plt.subplots(2, 2, figsize=(7.15, 4.35), sharex=True)
    for ax, (base, title) in zip(axes.ravel(), pairs):
        for law, label, colour, line_style in (("k0", r"$\kappa_0$", "#2166ac", "-"), ("kd", r"$\kappa(d)$", "#b2182b", "--")):
            t, value = state_history(f"{base}_{law}", "fracture_surface", 0.0)
            ax.plot(t, value, color=colour, ls=line_style, label=label)
        ax.set(ylabel=r"$\Gamma_\ell$ [mm]", title=title, xlim=(0.0, 0.1))
    axes[1, 0].set_xlabel(r"$t$ [s]")
    axes[1, 1].set_xlabel(r"$t$ [s]")
    axes[0, 0].legend(ncol=2, loc="lower center", bbox_to_anchor=(1.13, 1.17))
    panel_labels(axes)
    save(fig, "manuscript_controls.pdf", rect=(0.0, 0.025, 1.0, 1.0), h_pad=3.2)


def _profile_envelope(name: str, depth_multiple: float = 5.0):
    _, profiles = boundary_paths(name, depth_multiple)
    common = np.linspace(0.0, depth_multiple, 241)
    values = np.vstack([np.interp(common, coord, damage) for coord, damage in profiles])
    return common, np.percentile(values, 5, axis=0), np.median(values, axis=0), np.percentile(values, 95, axis=0)


def plot_boundary_statistics(resistance):
    fig, axes = plt.subplots(2, 2, figsize=(7.15, 4.85))
    colours = {100: "#2166ac", 200: "#ef8a62", 400: "#b2182b"}
    line_styles = {100: "-", 200: "--", 400: "-."}
    for nx in (100, 200, 400):
        name = f"TS_{nx:03d}x{int(nx * 0.4):03d}_n040_kd"
        n, p05, median, p95 = _profile_envelope(name)
        label = rf"$\ell/h={nx / 25:.0f}$"
        axes[0, 0].plot(n, median, color=colours[nx], ls=line_styles[nx], label=label)
        axes[0, 0].fill_between(n, p05, p95, color=colours[nx], alpha=0.10, linewidth=0)
        g05 = KRES + (1.0 - KRES) * (1.0 - p95) ** 2
        g50 = KRES + (1.0 - KRES) * (1.0 - median) ** 2
        g95 = KRES + (1.0 - KRES) * (1.0 - p05) ** 2
        axes[0, 1].semilogy(n, g50, color=colours[nx], ls=line_styles[nx])
        axes[0, 1].fill_between(n, g05, g95, color=colours[nx], alpha=0.10, linewidth=0)
    nref = np.linspace(0.0, 5.0, 300)
    dref = np.exp(-nref)
    axes[0, 0].plot(nref, dref, "k--", lw=1.0, label="one-sided interior-profile reference")
    axes[0, 1].semilogy(nref, KRES + (1.0 - KRES) * (1.0 - dref) ** 2, "k--", lw=1.0)
    axes[0, 0].set(xlabel=r"$\xi/\ell$", ylabel=r"$d(\xi)$", xlim=(0, 5), ylim=(0, 1.05))
    axes[0, 1].set(xlabel=r"$\xi/\ell$", ylabel=r"$\kappa(d)/\kappa_0$", xlim=(0, 5))
    axes[0, 0].legend(loc="upper right", ncol=1)

    paths5, _ = boundary_paths("TS_200x080_n080_kd", 5)
    box = axes[1, 0].boxplot(
        [paths5[edge] for edge in ("left", "bottom", "top")],
        tick_labels=["left", "bottom", "top"],
        whis=(5, 95),
        showfliers=False,
        patch_artist=True,
        medianprops={"color": "black", "linewidth": 1.1},
    )
    for patch, colour in zip(box["boxes"], ("#67a9cf", "#ef8a62", "#b2182b")):
        patch.set_facecolor(colour)
        patch.set_alpha(0.55)
    axes[1, 0].set(xlabel="cooled edge", ylabel=r"$R_{\rm layer}(s)$ [mm$^2$K/W]")

    depth = np.array([3.0, 4.0, 5.0])
    styles = (("all paths", None, "k", "o"), ("left", "left", "#67a9cf", "s"), ("bottom", "bottom", "#ef8a62", "^"), ("top", "top", "#b2182b", "D"))
    for label, edge, colour, marker in styles:
        values = []
        for number in (3, 4, 5):
            item = resistance["depths"][str(number)]
            values.append(item["mean"] if edge is None else item["edges"][edge]["mean"])
        axes[1, 1].plot(depth, values, marker=marker, color=colour, label=label)
    axes[1, 1].set(xlabel=r"integration depth [$\ell$]", ylabel=r"mean $R_{\rm layer}$ [mm$^2$K/W]", xticks=depth)
    axes[1, 1].legend(loc="best", ncol=2)
    panel_labels(axes)
    save(fig, "manuscript_boundary_statistics.pdf", rect=(0.0, 0.025, 1.0, 1.0), h_pad=3.2)


def plot_refinement(primary):
    selected = [row for row in primary if row["steps"] == 40]
    fig, axes = plt.subplots(2, 2, figsize=(7.15, 4.35))
    axes = axes.ravel()
    for law, label, colour, marker in (("k0", r"$\kappa_0$", "#2166ac", "o"), ("kd", r"$\kappa(d)$", "#b2182b", "s")):
        data = sorted((row for row in selected if row["law"] == law), key=lambda row: int(row["mesh"].split("x")[0]))
        ratio = np.asarray([int(row["mesh"].split("x")[0]) / 25.0 for row in data])
        axes[0].plot(ratio, [row["Q_out_J"] for row in data], marker=marker, color=colour, label=label)
        if law == "kd":
            axes[1].plot(ratio, [row["R_layer"] for row in data], marker=marker, color=colour)
        axes[2].plot(ratio, [row["T_core_K"] for row in data], marker=marker, color=colour)
        axes[3].plot(ratio, [row["Gamma_mm"] for row in data], marker=marker, color=colour)
    axes[1].axhline(0.5 * 546.2954177582037, color="k", ls="--", lw=0.9, label="one-sided reference")
    labels = ((r"$Q_{\rm out}$ [J]",), (r"$\overline{R}_{\rm layer}$ [mm$^2$K/W]",), (r"$\overline{T}_{\rm core}$ [K]",), (r"$\Gamma_\ell$ [mm]",))
    for ax, (ylabel,) in zip(axes, labels):
        ax.set(xlabel=r"$\ell/h$", ylabel=ylabel, xticks=[4, 8, 16])
    axes[0].legend(loc="best")
    axes[1].legend(loc="best")
    axes[1].annotate("15.4%", xy=(16, 207.04), xytext=(9.3, 225), arrowprops={"arrowstyle": "->", "lw": 0.7})
    panel_labels(axes)
    save(fig, "manuscript_refinement.pdf", rect=(0.0, 0.025, 1.0, 1.0), w_pad=1.0, h_pad=3.2)


def plot_profile_transport():
    profile = np.genfromtxt(ROOT / "results" / "verification" / "profile_transport" / "profile_convergence.csv", delimiter=",", names=True)
    flux = np.genfromtxt(ROOT / "results" / "verification" / "profile_transport" / "transport_convergence.csv", delimiter=",", names=True)
    fig, axes = plt.subplots(2, 2, figsize=(7.15, 5.25))
    x = np.linspace(-4.0, 4.0, 801)
    axes[0, 0].plot(x, np.exp(-np.abs(x)), "k-", label="exact")
    profile_styles = {
        2: ("#2166ac", "--", "o"),
        4: ("#ef8a62", "-.", "s"),
        8: ("#b2182b", ":", "^"),
    }
    for ratio, (colour, line_style, marker) in profile_styles.items():
        nodes = np.arange(-10.0, 10.0 + 1.0 / ratio, 1.0 / ratio)
        axes[0, 0].plot(
            x,
            np.interp(x, nodes, np.exp(-np.abs(nodes))),
            color=colour,
            ls=line_style,
            marker=marker,
            markevery=max(1, int(100 / ratio)),
            ms=2.4,
            mfc="white",
            mew=0.6,
            label=rf"$\ell/h={ratio}$",
        )
    axes[0, 0].set(xlabel=r"$x/\ell$", ylabel=r"prescribed $d(x)$", xlim=(-4, 4), ylim=(0, 1.03))
    axes[0, 0].legend(ncol=4, loc="lower center", bbox_to_anchor=(0.5, 1.075))

    ratio = profile["ell_over_h"]
    axes[0, 1].loglog(ratio, profile["profile_relative_L2"], "o-", color="#2166ac", label=r"profile $L_2$")
    axes[0, 1].loglog(ratio, profile["Gamma_relative_error"], "s--", color="#b2182b", label=r"$\Gamma_\ell$")
    axes[0, 1].set(xlabel=r"$\ell/h$", ylabel="relative error", xticks=[2, 4, 8, 16, 32, 64, 128])
    axes[0, 1].get_xaxis().set_major_formatter(mpl.ticker.ScalarFormatter())
    axes[0, 1].legend(loc="best")

    colours = {1e-2: "#2166ac", 1e-3: "#ef8a62", 1e-4: "#b2182b"}
    markers = {1e-2: "o", 1e-3: "s", 1e-4: "^"}
    for kres in (1e-2, 1e-3, 1e-4):
        subset = flux[np.isclose(flux["k_res"], kres)]
        axes[1, 0].loglog(subset["ell_over_h"], 100.0 * subset["heat_flux_relative_error"], marker=markers[kres], color=colours[kres], label=rf"$k_\kappa=10^{{{int(np.log10(kres))}}}$")
    for level, line_style in ((1.0, ":"), (5.0, "-."), (10.0, "--")):
        axes[1, 0].axhline(level, color="0.45", ls=line_style, lw=0.75)
        axes[1, 0].text(2.1, level * 1.08, f"{level:.0f}%", fontsize=6.5, color="0.35")
    axes[1, 0].set(xlabel=r"$\ell/h$", ylabel="heat-flux error [%]", xticks=[2, 4, 8, 16, 32, 64, 128])
    axes[1, 0].get_xaxis().set_major_formatter(mpl.ticker.ScalarFormatter())
    axes[1, 0].legend(ncol=3, loc="lower center", bbox_to_anchor=(0.5, 1.075))

    k = np.logspace(-5, 0, 400)
    exact = 2.0 * np.sqrt((1.0 - k) / k) * np.arctan(np.sqrt((1.0 - k) / k)) + np.log(1.0 / k)
    asymptote = np.pi / np.sqrt(k) + np.log(1.0 / k) - 2.0
    axes[1, 1].loglog(k, exact, "k-", label=r"exact $I(k_\kappa)$")
    axes[1, 1].loglog(k, asymptote, "--", color="#b2182b", label=r"small-$k_\kappa$ asymptote")
    axes[1, 1].axvline(KRES, color="#2166ac", ls=":", lw=0.9)
    axes[1, 1].set(xlabel=r"$k_\kappa$", ylabel=r"$I(k_\kappa)$")
    axes[1, 1].legend(loc="best")
    panel_labels(axes)
    save(fig, "manuscript_profile_transport.pdf", rect=(0.0, 0.025, 1.0, 1.0), h_pad=4.0)


def plot_transport_resolution():
    flux = np.genfromtxt(
        ROOT / "results" / "verification" / "profile_transport" / "transport_convergence.csv",
        delimiter=",",
        names=True,
    )
    fig, axes = plt.subplots(1, 2, figsize=(7.15, 3.05))
    colours = {1e-2: "#2166ac", 1e-3: "#ef8a62", 1e-4: "#b2182b"}
    markers = {1e-2: "o", 1e-3: "s", 1e-4: "^"}
    for kres in (1e-2, 1e-3, 1e-4):
        subset = flux[np.isclose(flux["k_res"], kres)]
        error = 100.0 * subset["heat_flux_relative_error"]
        label = rf"$k_\kappa=10^{{{int(np.log10(kres))}}}$"
        axes[0].loglog(
            subset["ell_over_h"], error, marker=markers[kres],
            color=colours[kres], label=label,
        )
        nth = subset["ell_over_h"] * np.sqrt(kres)
        axes[1].loglog(
            nth, error, marker=markers[kres], color=colours[kres], label=label,
        )
    for ax in axes:
        for level, line_style in ((1.0, ":"), (5.0, "-.")):
            ax.axhline(level, color="0.45", ls=line_style, lw=0.75)
        ax.set_ylabel("heat-flux error [%]")
        ax.legend(loc="best")
    axes[0].set_xlabel(r"phase-field resolution $N_{\rm pf}=\ell/h$")
    axes[1].set_xlabel(r"transport resolution $N_{\rm th}=\ell\sqrt{k_\kappa}/h$")
    axes[1].axvspan(0.64, 0.91, color="#fddbc7", alpha=0.35, lw=0)
    axes[1].axvspan(1.36, 1.75, color="#d1e5f0", alpha=0.35, lw=0)
    axes[1].text(0.67, 6.5, "5% threshold", fontsize=6.6, rotation=90, va="bottom")
    axes[1].text(1.41, 1.25, "1% threshold", fontsize=6.6, rotation=90, va="bottom")
    panel_labels(axes)
    save(fig, "manuscript_transport_resolution.pdf", rect=(0.0, 0.025, 1.0, 1.0), w_pad=1.4)


def _read_image(name):
    return plt.imread(FIG / name)


def _field_crop(name):
    """Crop the high-resolution ParaView export to the physical field."""
    image = _read_image(name)
    return image[115:985, 413:2587]


def _colourbar_crop(name):
    """Crop one publication-scale colour bar from a ParaView export."""
    image = _read_image(name)
    return image[90:1010, 2620:2975]


def _vector_colourbar(ax, cmap, limits, label, ticks):
    """Draw a vector colour bar with journal-scale typography."""
    scalar = mpl.cm.ScalarMappable(norm=Normalize(*limits), cmap=cmap)
    scalar.set_array([])
    bar = plt.colorbar(scalar, cax=ax, orientation="vertical", ticks=ticks)
    bar.set_label(label, fontsize=8.5, labelpad=3)
    bar.ax.tick_params(labelsize=7.5, direction="in", length=3, width=0.7)
    bar.outline.set_linewidth(0.6)
    return bar


def _save_field_figure(fig, stem):
    """Keep the ParaView raster above 600 ppi and retain vector annotations."""
    for suffix in ("pdf", "png"):
        fig.savefig(FIG / f"{stem}.{suffix}", bbox_inches="tight", pad_inches=0.03, dpi=600)
    plt.close(fig)


def plot_main_fields():
    files = (
        ("manuscript_main_k0_T.png", "manuscript_main_kd_T.png"),
        ("manuscript_main_k0_q.png", "manuscript_main_kd_q.png"),
        ("manuscript_main_k0_d.png", "manuscript_main_kd_d.png"),
    )
    fig = plt.figure(figsize=(7.15, 4.95))
    grid = GridSpec(3, 3, figure=fig, width_ratios=(1, 1, 0.075), wspace=0.08, hspace=0.48)
    axes = np.array([[fig.add_subplot(grid[i, j]) for j in range(2)] for i in range(3)])
    colour_axes = [fig.add_subplot(grid[i, 2]) for i in range(3)]
    for ax, name in zip(axes.ravel(), sum(files, ())):
        ax.imshow(_field_crop(name), interpolation="none")
        ax.set_axis_off()
    _vector_colourbar(colour_axes[0], TEMPERATURE_CMAP, (293.15, 573.15), r"$T$ [K]", [293.15, 350, 400, 450, 500, 573.15])
    _vector_colourbar(colour_axes[1], mpl.colormaps["magma"], (0.0, 5.25), r"$|\mathbf{q}|$ [W/mm$^2$]", [0, 1, 2, 3, 4, 5])
    _vector_colourbar(colour_axes[2], DAMAGE_CMAP, (0.0, 1.0), r"$d$", np.linspace(0.0, 1.0, 6))
    axes[0, 0].set_title(r"constant conductivity $\kappa_0$")
    axes[0, 1].set_title(r"damage-dependent conductivity $\kappa(d)$")
    axes[0, 0].text(-0.035, 0.5, "temperature", transform=axes[0, 0].transAxes, rotation=90, va="center", ha="right", fontweight="bold")
    axes[1, 0].text(-0.035, 0.5, "heat flux", transform=axes[1, 0].transAxes, rotation=90, va="center", ha="right", fontweight="bold")
    axes[2, 0].text(-0.035, 0.5, "damage", transform=axes[2, 0].transAxes, rotation=90, va="center", ha="right", fontweight="bold")
    panel_labels(axes, image_panels=True)
    _save_field_figure(fig, "main_fields")


def plot_mesh_fields():
    """Damage fields with fixed-size, high-contrast mesh zooms.

    A full 400x160 grid cannot remain legible at one-column print scale.  The
    upper row therefore retains the complete ParaView field and marks one
    common physical region; the lower row enlarges that region and overlays
    the exact structured Q1 grid using a dark/white halo that remains visible
    over the complete damage palette.
    """
    fig = plt.figure(figsize=(7.15, 2.65))
    grid = GridSpec(
        2, 4, figure=fig,
        width_ratios=(1, 1, 1, 0.065),
        height_ratios=(1.0, 1.0),
        wspace=0.10, hspace=0.12,
        left=0.045, right=0.94, bottom=0.11, top=0.89,
    )
    full_axes = [fig.add_subplot(grid[0, j]) for j in range(3)]
    zoom_axes = [fig.add_subplot(grid[1, j]) for j in range(3)]
    # The 1.25 x 0.50 mm window retains the plate aspect ratio.  Equal
    # GridSpec row heights therefore give identical panel dimensions while
    # limiting the finest enlargement to 40 x 16 cells.
    x0, x1 = 0.36, 0.46
    y0, y1 = 0.00, 0.10

    for col, (nx, full_ax, zoom_ax) in enumerate(zip((100, 200, 400), full_axes, zoom_axes)):
        ny = int(nx * 0.4)
        image = _field_crop(f"manuscript_mesh_{nx}_kd_d.png")
        height, width = image.shape[:2]
        ix0, ix1 = int(x0 * width), int(x1 * width)
        iy0, iy1 = int(y0 * height), int(y1 * height)
        zoom = image[iy0:iy1, ix0:ix1]

        full_ax.imshow(image, interpolation="none")
        full_ax.add_patch(Rectangle(
            (ix0, iy0), ix1 - ix0, iy1 - iy0,
            fill=False, edgecolor="#00d4ff", linewidth=1.35,
        ))
        full_ax.set_title(
            rf"${nx}\times{ny}$; {nx * ny:,} cells; $\ell/h={nx / 25:.0f}$",
            pad=5.0,
        )
        full_ax.set_axis_off()

        zoom_ax.imshow(zoom, interpolation="none")
        n_x_lines = int(round(nx * (x1 - x0)))
        n_y_lines = int(round(ny * (y1 - y0)))
        x_lines = np.linspace(-0.5, zoom.shape[1] - 0.5, n_x_lines + 1)
        y_lines = np.linspace(-0.5, zoom.shape[0] - 0.5, n_y_lines + 1)
        # A dark halo plus a thin white core keeps every edge visible on both
        # pale intact regions and dark damaged regions.
        for value in x_lines:
            zoom_ax.axvline(value, color="black", linewidth=0.62, alpha=0.86, zorder=3)
            zoom_ax.axvline(value, color="white", linewidth=0.20, alpha=0.95, zorder=4)
        for value in y_lines:
            zoom_ax.axhline(value, color="black", linewidth=0.62, alpha=0.86, zorder=3)
            zoom_ax.axhline(value, color="white", linewidth=0.20, alpha=0.95, zorder=4)
        zoom_ax.set_axis_off()
    colour_ax = fig.add_subplot(grid[:, 3])
    _vector_colourbar(colour_ax, DAMAGE_CMAP, (0.0, 1.0), r"$d$", np.linspace(0.0, 1.0, 6))
    panel_labels(full_axes + zoom_axes, image_panels=True, label_y=-0.07)
    _save_field_figure(fig, "mesh_fields")


def plot_control_fields():
    files = (
        ("manuscript_bi10_k0_T.png", "manuscript_bi10_kd_T.png"),
        ("manuscript_free_k0_d.png", "manuscript_free_kd_d.png"),
        ("manuscript_at1_k0_d.png", "manuscript_at1_kd_d.png"),
    )
    fig = plt.figure(figsize=(7.15, 5.25))
    grid = GridSpec(3, 3, figure=fig, width_ratios=(1, 1, 0.075), wspace=0.08, hspace=0.48)
    axes = np.array([[fig.add_subplot(grid[i, j]) for j in range(2)] for i in range(3)])
    colour_axes = [fig.add_subplot(grid[i, 2]) for i in range(3)]
    for ax, name in zip(axes.ravel(), sum(files, ())):
        ax.imshow(_field_crop(name), interpolation="none")
        ax.set_axis_off()
    _vector_colourbar(colour_axes[0], TEMPERATURE_CMAP, (293.15, 573.15), r"$T$ [K]", [293.15, 350, 400, 450, 500, 573.15])
    _vector_colourbar(colour_axes[1], DAMAGE_CMAP, (0.0, 1.0), r"$d$", np.linspace(0.0, 1.0, 6))
    _vector_colourbar(colour_axes[2], DAMAGE_CMAP, (0.0, 1.0), r"$d$", np.linspace(0.0, 1.0, 6))
    axes[0, 0].set_title(r"$\kappa_0$")
    axes[0, 1].set_title(r"$\kappa(d)$")
    for ax, label in zip(axes[:, 0], (r"$\mathrm{Bi}_0=10$: $T$", "minimal restraint: $d$", "AT1: $d$")):
        ax.text(-0.03, 0.5, label, transform=ax.transAxes, rotation=90, va="center", ha="right", fontweight="bold")
    panel_labels(axes, image_panels=True)
    _save_field_figure(fig, "control_fields")


def main():
    FIG.mkdir(parents=True, exist_ok=True)
    POST.mkdir(parents=True, exist_ok=True)
    TABLES.mkdir(parents=True, exist_ok=True)
    primary = collect(primary_names())
    controls_names = [f"TS_{kind}_n080_{law}" for kind in ("Bi10", "free", "gcT", "AT1") for law in ("k0", "kd")]
    controls = collect(controls_names)
    resistance = resistance_summary("TS_200x080_n080_kd")
    (POST / "boundary_resistance_statistics.json").write_text(json.dumps(resistance, indent=2) + "\n")
    (POST / "primary_summary.json").write_text(json.dumps(primary, indent=2) + "\n")
    (POST / "controls_summary.json").write_text(json.dumps(controls, indent=2) + "\n")
    write_tables_and_macros(primary, controls, resistance)
    plot_mechanism()
    plot_controls()
    plot_boundary_statistics(resistance)
    plot_refinement(primary)
    plot_profile_transport()
    plot_transport_resolution()
    plot_main_fields()
    plot_mesh_fields()
    plot_control_fields()
    print(json.dumps(resistance, indent=2))


if __name__ == "__main__":
    main()
