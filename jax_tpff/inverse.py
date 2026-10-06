"""Outer identification of Gc(T) coefficients from a quenching snapshot.

The inner forward solve is the staggered quench driver. The outer loop is
L-BFGS on scaled θ=(Gc0, b1, b2) with a finite-difference gradient:
UMFPACK is not differentiated through in this CPU campaign.
"""

from __future__ import annotations

import json
import shutil
import time
from dataclasses import replace
from pathlib import Path

import numpy as onp

from jax_tpff.cases import CaseSpec
from jax_tpff.lbfgs import _two_loop
from jax_tpff.staggered import _mesh, run_quench


def _obs_vec(sol_d, records):
    d = onp.asarray(sol_d).ravel()
    ncr = onp.array([r.get("n_cracked", 0) for r in records], dtype=float)
    dmean = onp.array([r.get("d_mean", 0.0) for r in records], dtype=float)
    nnode = max(len(d), 1)
    return onp.concatenate([d, ncr / nnode, dmean])


def _theta_mat(case: CaseSpec, theta):
    return replace(case.mat, Gc0=float(theta[0]), b1=float(theta[1]), b2=float(theta[2]))


def _run_forward(case: CaseSpec, theta, work_dir: Path):
    if work_dir.exists():
        shutil.rmtree(work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)
    fwd = replace(
        case,
        name=f"{case.name}_fwd",
        mode="quench",
        mat=_theta_mat(case, theta),
        solver="umfpack",
        save_every=max(int(case.save_every), 50),
    )
    records = run_quench(fwd, work_dir)
    ckpt = work_dir / "checkpoint.npz"
    data = onp.load(ckpt, allow_pickle=True)
    sol_d = onp.array(data["sol_d"])
    mesh = _mesh(fwd)
    last = records[-1] if records else {}
    return mesh.points, sol_d, last, records


def run_inverse(case: CaseSpec, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    theta_true = onp.array(
        case.theta_true if case.theta_true is not None else (case.mat.Gc0, case.mat.b1, case.mat.b2),
        dtype=float,
    )
    scale = onp.array([max(abs(theta_true[0]), 1e-8), 1.0, 1.0])
    y_true = theta_true / scale
    y_prior = onp.array([y_true[0], 0.0, 0.0], dtype=float)
    t0 = time.time()

    pts_true, d_true, last_true, rec_true = _run_forward(case, theta_true, out_dir / "obs")
    obs = _obs_vec(d_true, rec_true)
    onp.savez_compressed(
        out_dir / "obs_snapshot.npz", points=pts_true, sol_d=d_true, obs=obs, theta_true=theta_true
    )

    cache: dict[tuple, tuple] = {}
    n_fwd = [1]
    history = []

    def _key(y):
        return tuple(onp.round(onp.asarray(y, dtype=float), 6))

    def misfit_y(y):
        y = onp.asarray(y, dtype=float)
        k = _key(y)
        if k in cache:
            return cache[k]
        theta = y * scale
        try:
            _, sol_d, last, recs = _run_forward(case, theta, out_dir / "fwd")
            pred = _obs_vec(sol_d, recs)
            n = min(len(pred), len(obs))
            j_data = 0.5 * float(onp.mean((pred[:n] - obs[:n]) ** 2))
            j_reg = 0.5 * 1.0e-3 * float(onp.sum((y - y_prior) ** 2))
            j = j_data + j_reg
            rec = {
                "J": j,
                "J_data": j_data,
                "J_reg": j_reg,
                "n_cracked": last.get("n_cracked"),
                "d_max": last.get("d_max"),
            }
        except Exception as exc:
            print(f"[{case.name}] forward failed at theta={theta}: {type(exc).__name__}: {exc}", flush=True)
            j = 1.0e6
            rec = {"J": j, "J_data": j, "J_reg": 0.0, "error": str(exc)}
        n_fwd[0] += 1
        cache[k] = (j, rec, y.copy())
        return cache[k]

    def value_and_grad(y):
        j0, rec0, _ = misfit_y(y)
        g = onp.zeros(3)
        hvec = onp.array([5.0e-3, 8.0e-2, 8.0e-2])
        for i in range(3):
            yp = y.copy()
            yp[i] = yp[i] + hvec[i]
            jp, _, _ = misfit_y(yp)
            g[i] = (jp - j0) / hvec[i]
        return j0, g, rec0

    lo = onp.array([0.2, 0.0, 0.0])
    hi = onp.array([5.0, 4.0, 3.0])
    y = y_prior.copy()
    s_hist, y_hist = [], []
    maxiter = int(getattr(case, "lbfgs_maxiter", 4))
    j, g, rec = value_and_grad(y)
    print(f"[{case.name}] outer 000  J={j:.4e}  theta={y * scale}  g={g}  n_fwd={n_fwd[0]}", flush=True)
    history.append(
        {
            "iter": 0,
            "J": j,
            "theta": (y * scale).tolist(),
            **{k: rec.get(k) for k in ("J_data", "J_reg", "n_cracked", "d_max")},
        }
    )

    for it in range(maxiter):
        g_use = g.copy()
        at_lo = (y <= lo + 1e-16) & (g_use > 0)
        at_hi = (y >= hi - 1e-16) & (g_use < 0)
        g_use[at_lo | at_hi] = 0.0
        p = -_two_loop(g_use, s_hist, y_hist)
        p[at_lo | at_hi] = 0.0
        pn = float(onp.linalg.norm(p))
        if pn < 1e-16:
            break
        p = p * min(1.0, 0.75 / pn)
        alpha = 1.0
        accepted = False
        j_best, y_best = j, y
        for _ in range(6):
            y1 = onp.clip(y + alpha * p, lo, hi)
            j1, rec1, _ = misfit_y(y1)
            if j1 < j_best:
                j_best, y_best = j1, y1
            if j1 < j - 1e-16:
                j1, g1, rec1 = value_and_grad(y1)
                s = y1 - y
                yy = g1 - g
                if onp.dot(s, yy) > 1e-16:
                    s_hist.append(s)
                    y_hist.append(yy)
                    if len(s_hist) > 5:
                        s_hist.pop(0)
                        y_hist.pop(0)
                y, j, g, rec = y1, j1, g1, rec1
                accepted = True
                break
            alpha *= 0.5
        if not accepted:
            if j_best < j - 1e-16:
                y = y_best
                j, g, rec = value_and_grad(y)
            else:
                break
        history.append(
            {
                "iter": it + 1,
                "J": j,
                "theta": (y * scale).tolist(),
                "alpha": alpha,
                **{k: rec.get(k) for k in ("J_data", "J_reg", "n_cracked", "d_max")},
            }
        )
        print(f"[{case.name}] outer {it+1:03d}  J={j:.4e}  theta={y * scale}  n_fwd={n_fwd[0]}", flush=True)

    theta = y * scale
    rel = onp.abs(theta - theta_true) / onp.maximum(onp.abs(theta_true), 1e-12)
    wall = time.time() - t0
    n_node = int(len(pts_true))
    summary = {
        "case": case.name,
        "mode": "inverse",
        "solver": "outer-lbfgs-fd",
        "Nx": case.Nx,
        "Ny": case.Ny,
        "ell": case.mat.ell,
        "Gc0": case.mat.Gc0,
        "model": case.mat.model,
        "n_steps": last_true.get("step"),
        "wall_s": wall,
        "peak_force": None,
        "n_dof": n_node * 4,
        "n_node": n_node,
        "n_fwd": n_fwd[0],
        "theta_true": theta_true.tolist(),
        "theta_hat": theta.tolist(),
        "theta_prior": (y_prior * scale).tolist(),
        "rel_err_Gc0": float(rel[0]),
        "rel_err_b1": float(rel[1]),
        "rel_err_b2": float(rel[2]),
        "J": j,
        "last": last_true,
        "history": history,
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    if history:
        keys = ["iter", "J", "alpha", "J_data", "J_reg", "n_cracked", "d_max"]
        with open(out_dir / "history.csv", "w") as f:
            f.write(",".join(keys + ["Gc0", "b1", "b2"]) + "\n")
            for h in history:
                th = h.get("theta", [None, None, None])
                row = [h.get(k, "") for k in keys] + list(th)
                f.write(",".join(str(v) for v in row) + "\n")
    print(
        f"[{case.name}] done  rel_err=({rel[0]:.3e}, {rel[1]:.3e}, {rel[2]:.3e})  "
        f"n_fwd={n_fwd[0]}  wall={wall:.1f}s",
        flush=True,
    )
    return history
