#!/usr/bin/env python3
"""One-dimensional AT2 profile and diffuse-layer transport verification."""

from __future__ import annotations

import csv
import json
import os
import shutil
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(__file__).resolve().parents[1] / ".mplconfig"))

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import quad
from scipy.optimize import brentq
from scipy.sparse import lil_matrix
from scipy.sparse.linalg import spsolve


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "verification" / "profile_transport"
FIG = ROOT / "paper_artifacts" / "figures"

mpl.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
    "mathtext.fontset": "stix",
    "font.size": 9,
    "axes.labelsize": 9,
    "legend.fontsize": 7.5,
    "xtick.direction": "in",
    "ytick.direction": "in",
    "xtick.top": False,
    "ytick.right": False,
    "axes.spines.top": True,
    "axes.spines.right": True,
    "pdf.fonttype": 42,
})


def resistance_factor(k: float) -> float:
    """Dimensionless excess resistance for d=exp(-|x|/ell)."""
    a = np.sqrt((1.0 - k) / k)
    return float(2.0 * a * np.arctan(a) + np.log(1.0 / k))


def target_k(ell: float, target_R: float, kappa0: float) -> float:
    fn = lambda k: ell / kappa0 * resistance_factor(k) - target_R
    return float(brentq(fn, 1.0e-12, 1.0 - 1.0e-12))


def solve_profile(ratio: int, ell: float = 1.0, half_width_factor: float = 10.0):
    L = half_width_factor * ell
    n_elem = int(round(2.0 * half_width_factor * ratio))
    x = np.linspace(-L, L, n_elem + 1)
    h = x[1] - x[0]
    n = len(x)
    A = lil_matrix((n, n), dtype=float)
    for e in range(n_elem):
        ke = ell * ell / h * np.array([[1.0, -1.0], [-1.0, 1.0]])
        me = h / 6.0 * np.array([[2.0, 1.0], [1.0, 2.0]])
        nodes = (e, e + 1)
        A[np.ix_(nodes, nodes)] += ke + me
    prescribed = {0: 0.0, n // 2: 1.0, n - 1: 0.0}
    free = np.array([i for i in range(n) if i not in prescribed], dtype=int)
    fixed = np.array(sorted(prescribed), dtype=int)
    values = np.array([prescribed[i] for i in fixed])
    d = np.zeros(n)
    d[fixed] = values
    Af = A.tocsr()
    d[free] = spsolve(Af[free][:, free], -Af[free][:, fixed] @ values)

    d_exact = np.exp(-np.abs(x) / ell)
    l2 = np.sqrt(np.trapezoid((d - d_exact) ** 2, x) / np.trapezoid(d_exact**2, x))
    gamma = 0.0
    for e in range(n_elem):
        d0, d1 = d[e], d[e + 1]
        int_d2 = h / 3.0 * (d0 * d0 + d0 * d1 + d1 * d1)
        int_grad2 = (d1 - d0) ** 2 / h
        gamma += 0.5 * (int_d2 / ell + ell * int_grad2)
    return x, d, d_exact, float(l2), float(gamma)


def solve_transport(x: np.ndarray, d: np.ndarray, k_res: float = 1.0e-2, kappa0: float = 1.0):
    n = len(x)
    K = lil_matrix((n, n), dtype=float)
    gp = (-1.0 / np.sqrt(3.0), 1.0 / np.sqrt(3.0))
    for e in range(n - 1):
        h = x[e + 1] - x[e]
        k_int = 0.0
        for xi in gp:
            N0, N1 = 0.5 * (1.0 - xi), 0.5 * (1.0 + xi)
            dq = N0 * d[e] + N1 * d[e + 1]
            kval = kappa0 * (k_res + (1.0 - k_res) * (1.0 - dq) ** 2)
            k_int += kval * h / 2.0
        ke = k_int / (h * h) * np.array([[1.0, -1.0], [-1.0, 1.0]])
        K[np.ix_((e, e + 1), (e, e + 1))] += ke
    K = K.tocsr()
    fixed = np.array([0, n - 1])
    values = np.array([1.0, 0.0])
    free = np.arange(1, n - 1)
    T = np.zeros(n)
    T[fixed] = values
    T[free] = spsolve(K[free][:, free], -K[free][:, fixed] @ values)
    q_fe = float(abs((K @ T)[0]))

    ell = 1.0
    L = float(x[-1])
    integrand = lambda xx: 1.0 / (
        kappa0 * (k_res + (1.0 - k_res) * (1.0 - np.exp(-abs(xx) / ell)) ** 2)
    )
    R_exact = quad(integrand, -L, L, epsabs=1.0e-11, epsrel=1.0e-11, limit=400)[0]
    q_exact = 1.0 / R_exact
    return q_fe, float(q_exact), abs(q_fe - q_exact) / q_exact


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    FIG.mkdir(parents=True, exist_ok=True)
    ratios = [2, 4, 8, 16, 32, 64, 128]
    profile_rows = []
    transport_rows = []
    profiles = {}
    for ratio in ratios:
        x, d, exact, l2, gamma = solve_profile(ratio)
        profile_rows.append({
            "ell_over_h": ratio,
            "profile_relative_L2": l2,
            "Gamma_ell": gamma,
            "Gamma_relative_error": abs(gamma - 1.0),
        })
        for k_res in (1.0e-2, 1.0e-3, 1.0e-4):
            q_fe, q_exact, q_rel = solve_transport(x, d, k_res=k_res)
            transport_rows.append({
                "ell_over_h": ratio,
                "k_res": k_res,
                "heat_flux_FE": q_fe,
                "heat_flux_exact": q_exact,
                "heat_flux_relative_error": q_rel,
            })
        profiles[ratio] = (x, d, exact)

    with (OUT / "profile_convergence.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(profile_rows[0]))
        writer.writeheader()
        writer.writerows(profile_rows)
    with (OUT / "transport_convergence.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(transport_rows[0]))
        writer.writeheader()
        writer.writerows(transport_rows)

    kappa0 = 0.031
    ell0 = 0.5
    k0 = 1.0e-2
    target_R = ell0 / kappa0 * resistance_factor(k0)
    calibration = [
        {"ell_mm": ell, "k_res_for_fixed_R": target_k(ell, target_R, kappa0)}
        for ell in (0.25, 0.5, 1.0)
    ]
    payload = {
        "profile_convergence": profile_rows,
        "transport_convergence": transport_rows,
        "resistance_formula": "I(k)=2*sqrt((1-k)/k)*atan(sqrt((1-k)/k))+ln(1/k)",
        "baseline": {"ell_mm": ell0, "k_res": k0, "R_pf_mm2K_per_W": target_R},
        "fixed_resistance_calibration": calibration,
    }
    (OUT / "summary.json").write_text(json.dumps(payload, indent=2))

    fig, axs = plt.subplots(1, 4, figsize=(7.15, 2.20))
    x, d, exact = profiles[4]
    keep = np.abs(x) <= 4.0
    axs[0].plot(x[keep], exact[keep], "k-", lw=1.2, label="analytical")
    axs[0].plot(x[keep], d[keep], "o", ms=2.8, mfc="white", label=r"FE, $\ell/h=4$")
    axs[0].set(xlabel=r"$x/\ell$", ylabel=r"$d$")
    axs[0].legend(frameon=False, loc="lower left", fontsize=6.6)

    r = np.asarray(ratios, dtype=float)
    axs[1].plot(r, [row["profile_relative_L2"] for row in profile_rows], "o-", label=r"profile $L_2$")
    axs[1].plot(r, [row["Gamma_relative_error"] for row in profile_rows], "s-", label=r"$|\Gamma_\ell-1|$")
    axs[1].set_yscale("log")
    axs[1].set_xticks(r)
    axs[1].set_xscale("log", base=2)
    axs[1].set_xlim(1.8, 140)
    axs[1].set(xlabel=r"$\ell/h$", ylabel="relative error")
    axs[1].legend(frameon=False)

    for k_res, marker in ((1.0e-2, "o"), (1.0e-3, "s"), (1.0e-4, "^")):
        selected = [row for row in transport_rows if row["k_res"] == k_res]
        axs[2].plot(
            r,
            [row["heat_flux_relative_error"] for row in selected],
            marker + "-",
            label=rf"$k_\kappa=10^{{{int(np.log10(k_res))}}}$",
        )
    axs[2].set_xscale("log", base=2)
    axs[2].set_yscale("log")
    axs[2].set_xticks(r)
    axs[2].set_xlim(1.8, 140)
    axs[2].set(xlabel=r"$\ell/h$", ylabel="heat-flux error")
    axs[2].legend(frameon=False, fontsize=6.6)

    ks = np.logspace(-5, -1, 200)
    exact_I = np.array([resistance_factor(k) for k in ks])
    asym_I = np.pi / np.sqrt(ks)
    axs[3].loglog(ks, exact_I, "-", label="exact")
    small = ks <= 1.0e-3
    axs[3].loglog(ks[small], asym_I[small], "--", label=r"$\pi/\sqrt{k_\kappa}$")
    axs[3].axvline(k0, color="0.45", lw=0.8, ls=":")
    axs[3].set(xlabel=r"$k_\kappa$", ylabel=r"$\kappa_0 R_{\rm pf}/\ell$")
    axs[3].legend(frameon=False, fontsize=6.6)
    for i, (label, ax) in enumerate(zip(("(a)", "(b)", "(c)", "(d)"), axs)):
        xpos, ypos, valign = (0.02, 0.96, "top") if i == 0 else (0.02, 1.01, "bottom")
        ax.text(xpos, ypos, label, transform=ax.transAxes, va=valign, ha="left",
                bbox={"facecolor": "white", "edgecolor": "none", "pad": 0.7})
        ax.minorticks_on()
        ax.tick_params(which="both", direction="in", top=False, right=False)
        for spine in ax.spines.values():
            spine.set_visible(True)
    fig.tight_layout(w_pad=0.9)
    out_pdf = OUT / "profile_transport.pdf"
    fig.savefig(out_pdf, bbox_inches="tight", pad_inches=0.03)
    plt.close(fig)
    shutil.copyfile(out_pdf, FIG / "profile_transport.pdf")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
