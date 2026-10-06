#!/usr/bin/env python3
"""Publication plots and machine-readable metrics for the added validation cases."""

from __future__ import annotations

import csv
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/jax_tpff_mpl_validation")

import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np

FIG = ROOT / "paper_artifacts" / "figures"
FIG.mkdir(exist_ok=True)


def _summary(name: str) -> dict:
    return json.loads((ROOT / "results/verification" / name / "summary.json").read_text())


def _history(root: str, name: str) -> dict[str, np.ndarray]:
    with (ROOT / root / name / "history.csv").open() as stream:
        rows = list(csv.DictReader(stream))
    return {
        key: np.asarray([float(row[key]) for row in rows])
        for key in rows[0]
        if key != "converged"
    }


def _save(fig: plt.Figure, name: str) -> None:
    for ax in fig.axes:
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_linewidth(0.8)
        ax.minorticks_on()
        ax.tick_params(axis="both", which="major", direction="in", top=False, right=False, width=0.8, length=3.8)
        ax.tick_params(axis="both", which="minor", direction="in", top=False, right=False, width=0.7, length=2.1)
    fig.tight_layout(pad=0.45, w_pad=0.80)
    fig.savefig(FIG / name, bbox_inches="tight", pad_inches=0.03)
    plt.close(fig)


def _patch_exact(displacement: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Closed-form homogeneous AT2 solution for the affine plane-stress patch."""
    E, nu, gc, ell, residual = 210.0, 0.3, 2.7e-3, 0.04, 1.0e-6
    mu = E / (2.0 * (1.0 + nu))
    lam_3d = E * nu / ((1.0 + nu) * (1.0 - 2.0 * nu))
    lam = 2.0 * mu * lam_3d / (lam_3d + 2.0 * mu)
    strain = displacement  # Ly = 1 mm
    history = (0.5 * lam * (1.0 - nu) ** 2 + mu) * strain**2
    damage = 2.0 * history / (gc / ell + 2.0 * history)
    sigma_yy = ((1.0 - damage) ** 2 + residual) * (
        lam * (1.0 - nu) + 2.0 * mu
    ) * strain
    return damage, sigma_yy  # Lx = 1 mm, so reaction equals sigma_yy


def _maximum_principle(name: str) -> dict[str, float | int | bool]:
    h = _history("results/verification", name)
    overshoot = float(max(np.max(h["T_max"]) - 573.15, 0.0))
    undershoot = float(max(293.15 - np.min(h["T_min"]), 0.0))
    return {
        "violations": int(np.sum((h["T_max"] > 573.15 + 1.0e-8) | (h["T_min"] < 293.15 - 1.0e-8))),
        "maximum_overshoot_K": overshoot,
        "maximum_undershoot_K": undershoot,
        "passes": bool(overshoot <= 1.0e-8 and undershoot <= 1.0e-8),
    }


def main() -> None:
    mpl.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
            "mathtext.fontset": "stix",
            "font.size": 8.8,
            "axes.labelsize": 9.3,
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
            "legend.fontsize": 7.6,
            "lines.linewidth": 1.25,
            "lines.markersize": 3.8,
            "pdf.fonttype": 42,
        }
    )

    # Analytical homogeneous phase-field patch and four solver paths.
    patch_names = [
        "patch_staggered",
        "patch_jax",
        "patch_lbfgs",
        "patch_lbfgsb",
    ]
    patch_labels = ["sparse direct", "JAX Krylov", "block LMQN", "projected block LMQN"]
    patch_hist = [_history("results/verification", name) for name in patch_names]
    patch_summ = [_summary(name) for name in patch_names]
    u = patch_hist[0]["disp"]
    d_exact, f_exact = _patch_exact(u)

    fig, axs = plt.subplots(1, 3, figsize=(7.25, 2.45))
    axs[0].plot(u, f_exact, "k-", label="analytical")
    axs[1].plot(u, d_exact, "k-", label="analytical")
    styles = [("o", "#2166ac"), ("s", "#1b7837")]
    for idx, (marker, colour) in enumerate(styles):
        axs[0].plot(u, patch_hist[idx]["force"], marker, color=colour, ls="none", label=patch_labels[idx])
        axs[1].plot(u, patch_hist[idx]["d_mean"], marker, color=colour, ls="none", label=patch_labels[idx])
    axs[0].set(xlabel="prescribed displacement [mm]", ylabel="reaction [kN]")
    axs[1].set(xlabel="prescribed displacement [mm]", ylabel=r"mean damage $\overline{d}$")
    axs[0].legend(loc="lower center", bbox_to_anchor=(0.5, 1.01), ncol=3)
    axs[1].legend(loc="lower center", bbox_to_anchor=(0.5, 1.01), ncol=3)

    errors = np.asarray([max(float(row["last"]["err"]), 1.0e-16) for row in patch_summ])
    colours = ["#2166ac", "#1b7837", "#b2182b", "#ef8a62"]
    bars = axs[2].bar(np.arange(4), errors, color=colours, width=0.68)
    axs[2].axhline(1.0e-8, color="k", ls="--", lw=0.9, label=r"tolerance $10^{-8}$")
    axs[2].set_yscale("log")
    axs[2].set_ylabel("terminal increment measure")
    axs[2].set_xticks(np.arange(4), ["direct", "Krylov", "LMQN", "proj.\nLMQN"])
    axs[2].set_ylim(2.0e-17, 2.0e-1)
    axs[2].legend(loc="lower center", bbox_to_anchor=(0.5, 1.01))
    for bar, row in zip(bars, patch_summ):
        axs[2].text(
            bar.get_x() + bar.get_width() / 2.0,
            max(float(row["last"]["err"]), 2.0e-16) * 1.8,
            f'{row["wall_s"]:.1f}s',
            ha="center",
            va="bottom",
            fontsize=7.2,
            rotation=90 if float(row["last"]["err"]) < 1.0e-10 else 0,
        )
    for ax, tag in zip(axs, ["(a)", "(b)", "(c)"]):
        ax.text(0.02, 0.96, tag, transform=ax.transAxes, ha="left", va="top", fontweight="bold")
    _save(fig, "val_patch_solver.pdf")

    patch_metrics: dict[str, dict] = {}
    for name, label, h, row in zip(patch_names, patch_labels, patch_hist, patch_summ):
        force_scale = max(float(np.max(np.abs(f_exact))), 1.0e-30)
        patch_metrics[name] = {
            "label": label,
            "wall_s": float(row["wall_s"]),
            "terminal_increment_measure": float(row["last"]["err"]),
            "max_abs_damage_error": float(np.max(np.abs(h["d_mean"] - d_exact))),
            "max_relative_reaction_error": float(np.max(np.abs(h["force"] - f_exact)) / force_scale),
            "accepted": bool(float(row["last"]["err"]) <= 1.0e-8),
        }

    # Fixed-mesh backward-Euler refinement. The existing 10-step case
    # supplies the middle member without rerunning an identical input deck.
    time_specs = [
        (5, "results/verification", "T2_time_005"),
        (10, "outputs_p0", "T2_p0_at2_k0"),
        (20, "results/verification", "T2_time_020"),
        (40, "results/verification", "T2_time_040"),
        (80, "results/verification", "T2_time_080"),
    ]
    time_hist = [(n, _history(root, name)) for n, root, name in time_specs]
    fig, axs = plt.subplots(1, 3, figsize=(7.25, 2.45))
    palette = plt.cm.viridis(np.linspace(0.08, 0.9, len(time_hist)))
    for (n, h), colour in zip(time_hist, palette):
        axs[0].plot(h["time"], h["T_core_mean"], color=colour, marker="o", markevery=max(1, n // 10), label=f"{n} steps")
        axs[1].plot(h["time"], h["fracture_surface"], color=colour, marker="o", markevery=max(1, n // 10))
    axs[0].set(xlabel="time [s]", ylabel=r"$\overline{T}_{\rm core}$ [K]")
    axs[1].set(xlabel="time [s]", ylabel=r"$\Gamma_\ell$ [mm]")
    axs[0].legend(ncol=3, loc="lower center", bbox_to_anchor=(0.5, 1.01))

    terminal = {
        n: np.asarray([h["T_core_mean"][-1], h["fracture_surface"][-1], h["damage_area_d08"][-1]])
        for n, h in time_hist
    }
    ref = terminal[80]
    n_steps_axis = np.asarray([n for n, _ in time_hist[:-1]])
    rel = np.asarray([np.abs(terminal[n] - ref) / np.maximum(np.abs(ref), 1.0e-30) for n, _ in time_hist[:-1]])
    for idx, (label, marker) in enumerate([(r"$\overline{T}_{\rm core}$", "o"), (r"$\Gamma_\ell$", "s"), (r"$A_{d\geq0.8}$", "^")]):
        axs[2].semilogy(n_steps_axis, rel[:, idx], marker + "-", label=label)
    axs[2].set(xlabel="number of time steps", ylabel="relative difference from 80-step result")
    axs[2].set_xticks(n_steps_axis, [str(value) for value in n_steps_axis])
    axs[2].legend(loc="lower center", bbox_to_anchor=(0.5, 1.01), ncol=3)
    for ax, tag in zip(axs, ["(a)", "(b)", "(c)"]):
        ax.text(0.02, 0.96, tag, transform=ax.transAxes, ha="left", va="top", fontweight="bold")
    _save(fig, "val_time_convergence.pdf")

    # Accepted 20-step matched conductivity pair.
    h_k0 = _history("results/verification", "T2_time_020")
    h_kd = _history("results/verification", "T2_time_020_kd")
    fig, axs = plt.subplots(1, 3, figsize=(7.25, 2.45))
    for h, label, style in [(h_k0, r"$\kappa_0$", "-"), (h_kd, r"$\kappa(d)$", "--")]:
        axs[0].plot(h["time"], h["T_core_mean"], style, marker="o", label=label)
        axs[1].plot(h["time"], h["fracture_surface"], style, marker="o")
        axs[2].plot(h["time"], h["damage_area_d08"], style, marker="o")
    axs[0].set(xlabel="time [s]", ylabel=r"$\overline{T}_{\rm core}$ [K]")
    axs[1].set(xlabel="time [s]", ylabel=r"$\Gamma_\ell$ [mm]")
    axs[2].set(xlabel="time [s]", ylabel=r"$A_{d\geq0.8}$ [mm$^2$]")
    axs[0].legend(loc="lower center", bbox_to_anchor=(0.5, 1.01), ncol=2)
    for ax, tag in zip(axs, ["(a)", "(b)", "(c)"]):
        ax.text(0.02, 0.96, tag, transform=ax.transAxes, ha="left", va="top", fontweight="bold")
    _save(fig, "val_time_pair.pdf")

    time_rows = []
    for n, root, name in time_specs:
        row = json.loads((ROOT / root / name / "summary.json").read_text())
        last = row["last"]
        time_rows.append(
            {
                "steps": n,
                "dt_s": 0.1 / n,
                "wall_s": float(row["wall_s"]),
                "terminal_increment_measure": float(last["err"]),
                "T_max_K": float(last["T_max"]),
                "T_core_mean_K": float(last["T_core_mean"]),
                "fracture_surface_mm": float(last["fracture_surface"]),
                "damage_area_d08_mm2": float(last["damage_area_d08"]),
            }
        )

    k0 = _summary("T2_time_020")["last"]
    kd = _summary("T2_time_020_kd")["last"]
    metrics = {
        "analytical_patch": patch_metrics,
        "constant_conductivity_time_refinement": time_rows,
        "20_step_matched_pair": {
            "constant_conductivity": k0,
            "damage_dependent_conductivity": kd,
            "core_temperature_increase_K": float(kd["T_core_mean"] - k0["T_core_mean"]),
            "fracture_surface_reduction_percent": float(100.0 * (1.0 - kd["fracture_surface"] / k0["fracture_surface"])),
            "high_damage_area_reduction_percent": float(100.0 * (1.0 - kd["damage_area_d08"] / k0["damage_area_d08"])),
        },
        "maximum_principle_checks": {
            "20_step_constant": _maximum_principle("T2_time_020"),
            "20_step_degraded": _maximum_principle("T2_time_020_kd"),
            "40_step_constant": _maximum_principle("T2_time_040"),
            "40_step_degraded": _maximum_principle("T2_time_040_kd"),
        },
    }
    out = ROOT / "results/verification" / "validation_metrics.json"
    out.write_text(json.dumps(metrics, indent=2) + "\n")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
