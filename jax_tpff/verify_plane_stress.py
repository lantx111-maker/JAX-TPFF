#!/usr/bin/env python3
"""Verify the effective-Lame plane-stress reduction used by the 2-D model."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from jax_tpff.cases import ceramic_manuscript
from jax_tpff.constitutive import make_stress_fn


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "verification" / "plane_stress_verification"


def main():
    mat = ceramic_manuscript(0.5, False)
    mu = mat.mu
    lam3 = mat.E * mat.nu / ((1.0 + mat.nu) * (1.0 - 2.0 * mat.nu))
    lam2 = mat.lmbda
    rng = np.random.default_rng(20260923)
    max_sigma_zz = 0.0
    max_inplane_error = 0.0
    for _ in range(200):
        a = rng.normal(scale=2.0e-3, size=(2, 2))
        eps2 = 0.5 * (a + a.T)
        eps_zz = -lam3 * np.trace(eps2) / (lam3 + 2.0 * mu)
        eps3 = np.zeros((3, 3))
        eps3[:2, :2] = eps2
        eps3[2, 2] = eps_zz
        sig3 = lam3 * np.trace(eps3) * np.eye(3) + 2.0 * mu * eps3
        sig2 = lam2 * np.trace(eps2) * np.eye(2) + 2.0 * mu * eps2
        max_sigma_zz = max(max_sigma_zz, abs(sig3[2, 2]))
        max_inplane_error = max(max_inplane_error, float(np.max(np.abs(sig3[:2, :2] - sig2))))

    # The constitutive implementation adds a fixed diagonal strain perturbation
    # below 1e-10 only when differentiating the split stress.  Quantify its
    # rotational artefact at an exactly repeated tensile principal strain.
    stress_fn, _ = make_stress_fn(mat, 2)
    eps_iso = 1.0e-3 * np.eye(2)
    stress_iso = np.asarray(stress_fn(eps_iso, np.array([0.0]), mat.T_ref))
    max_repeat_rotation_error = 0.0
    max_repeat_deviator = 0.0
    for theta in np.linspace(0.0, 2.0 * np.pi, 73):
        c, s = np.cos(theta), np.sin(theta)
        rotation = np.array([[c, -s], [s, c]])
        rotated_back = rotation.T @ stress_iso @ rotation
        max_repeat_rotation_error = max(
            max_repeat_rotation_error,
            float(np.max(np.abs(rotated_back - stress_iso))),
        )
        mean_stress = 0.5 * np.trace(rotated_back)
        max_repeat_deviator = max(
            max_repeat_deviator,
            float(np.max(np.abs(rotated_back - mean_stress * np.eye(2)))),
        )
    payload = {
        "E_GPa": mat.E,
        "nu": mat.nu,
        "lambda_3d_GPa": lam3,
        "mu_GPa": mu,
        "lambda_plane_stress_GPa": lam2,
        "samples": 200,
        "max_abs_sigma_zz_GPa": max_sigma_zz,
        "max_abs_inplane_difference_GPa": max_inplane_error,
        "eigenvalue_perturbation_max_abs_strain": 1.0e-10,
        "repeated_eigenvalue_rotation_error_GPa": max_repeat_rotation_error,
        "repeated_eigenvalue_deviator_GPa": max_repeat_deviator,
        "scope": "undamaged isotropic plane-stress condensation; the split model uses the same effective two-dimensional parameter as an approximation",
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "summary.json").write_text(json.dumps(payload, indent=2))
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
