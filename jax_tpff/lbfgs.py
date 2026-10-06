"""Monolithic L-BFGS / L-BFGS-B on concatenated JAX-FEM residuals.

The residual of each subproblem is the variational gradient of the
corresponding energy. Concatenating R_u, R_d (and R_T) therefore supplies
the gradient that the two-loop recursion of Nocedal acts on. The initial
inverse Hessian is the block-diagonal tangent diag(K_uu, K_dd, K_TT)^-1,
so the first search direction is a staggered Newton step. History-field
irreversibility is used with unconstrained L-BFGS; L-BFGS-B drops the
history field and projects damage onto [d_n, 1].
"""

from __future__ import annotations

import numpy as onp
import scipy.sparse as sp
from scipy.sparse.linalg import splu

import jax
import jax.numpy as jnp
from jax_fem.solver import apply_bc_vec, assign_bc
import jax.flatten_util


class _DiagFactor:
    def __init__(self, diag):
        self.diag = onp.maximum(onp.asarray(diag, dtype=onp.float64), 1e-12)

    def solve(self, b):
        return onp.asarray(b, dtype=onp.float64) / self.diag


def _ravel(sol_list):
    return jnp.array(jax.flatten_util.ravel_pytree(sol_list)[0])


def _residual(problem, sol_list):
    res_list = problem.compute_residual(sol_list)
    dofs = _ravel(sol_list)
    res = apply_bc_vec(jax.flatten_util.ravel_pytree(res_list)[0], dofs, problem)
    return onp.array(res)


def _assign_bc_sol(problem, sol_list):
    dofs = assign_bc(_ravel(sol_list), problem)
    return problem.unflatten_fn_sol_list(dofs)


def _dirichlet_rows(problem):
    fe = problem.fes[0]
    rows = []
    for nodes, comps in zip(fe.node_inds_list, fe.vec_inds_list):
        rows.append(onp.asarray(nodes) * fe.vec + onp.asarray(comps))
    if not rows:
        return onp.array([], dtype=int)
    return onp.concatenate(rows).astype(int)


def _factor_problem(problem, sol_list):
    problem.newton_update(sol_list)
    n = int(problem.num_total_dofs_all_vars)
    V = onp.asarray(problem.V, dtype=onp.float64).ravel()
    I = onp.asarray(problem.I).ravel()
    J = onp.asarray(problem.J).ravel()
    m = int(min(len(V), len(I), len(J)))
    A = sp.csr_matrix((V[:m], (I[:m], J[:m])), shape=(n, n))
    A = A.tolil()
    rows = _dirichlet_rows(problem)
    if rows.size:
        A[rows, :] = 0.0
        A[rows, rows] = 1.0
    A = A.tocsc() + 1e-12 * sp.eye(n, format="csc")
    try:
        return splu(A)
    except Exception:
        return _DiagFactor(A.diagonal())


def _make_h0inv(pairs):
    factors = []
    sizes = []
    for problem, sol_list in pairs:
        factors.append(_factor_problem(problem, sol_list))
        sizes.append(int(problem.num_total_dofs_all_vars))

    def h0inv(q):
        out = []
        i = 0
        for fac, n in zip(factors, sizes):
            out.append(onp.asarray(fac.solve(onp.asarray(q[i : i + n], dtype=onp.float64))))
            i += n
        return onp.concatenate(out)

    return h0inv


def _dirichlet_free_mask(problem):
    fe = problem.fes[0]
    mask = onp.ones((fe.num_total_nodes, fe.vec), dtype=bool)
    for nodes, comps in zip(fe.node_inds_list, fe.vec_inds_list):
        mask[onp.array(nodes), onp.array(comps)] = False
    return mask.ravel()


def _two_loop(g, s_hist, y_hist, h0inv=None):
    q = g.copy()
    alphas = []
    for s, y in zip(reversed(s_hist), reversed(y_hist)):
        rho = 1.0 / (onp.dot(y, s) + 1e-30)
        a = rho * onp.dot(s, q)
        alphas.append(a)
        q = q - a * y
    if h0inv is not None:
        z = h0inv(q)
    elif y_hist:
        yy = onp.dot(y_hist[-1], y_hist[-1]) + 1e-30
        gamma = onp.dot(s_hist[-1], y_hist[-1]) / yy
        z = gamma * q
    else:
        z = q
    for s, y, a in zip(s_hist, y_hist, reversed(alphas)):
        rho = 1.0 / (onp.dot(y, s) + 1e-30)
        b = rho * onp.dot(y, z)
        z = z + s * (a - b)
    return z


def _line_search(x, p, rx, r_fn, max_back=8):
    n0 = float(onp.linalg.norm(rx))
    gdotp = float(onp.dot(rx, p))
    alpha = 1.0
    x_best, r_best, n_best = x, rx, n0
    for _ in range(max_back):
        x1 = x + alpha * p
        r1 = r_fn(x1)
        n1 = float(onp.linalg.norm(r1))
        if n1 < n_best:
            x_best, r_best, n_best = x1, r1, n1
        if n1 < n0 and (gdotp >= 0.0 or n1 <= n0 + 1e-4 * alpha * gdotp):
            return x1, r1, True
        alpha *= 0.5
    if n_best < n0:
        return x_best, r_best, True
    return x, rx, False


def lbfgs_minimize(r_fn, x0, free, m=20, maxiter=40, tol=1e-5, bounds=None, h0inv=None, refresh_h0=None, refresh_every=8):
    x = onp.array(x0, dtype=onp.float64)
    if bounds is not None:
        lo, hi = bounds
        x = onp.clip(x, lo, hi)
    rx = r_fn(x)
    s_hist, y_hist = [], []
    it = 0
    n0 = float(onp.linalg.norm(rx))
    for it in range(maxiter):
        nrm = float(onp.linalg.norm(rx))
        if nrm <= tol or (n0 > 0 and nrm / n0 <= 1e-6):
            break
        if refresh_h0 is not None and it > 0 and it % refresh_every == 0:
            h0inv = refresh_h0(x)
        g = rx.copy()
        g[~free] = 0.0
        at_lo = at_hi = None
        if bounds is not None:
            lo, hi = bounds
            at_lo = (x <= lo + 1e-14) & (g > 0)
            at_hi = (x >= hi - 1e-14) & (g < 0)
            g[at_lo | at_hi] = 0.0
        p = -_two_loop(g, s_hist, y_hist, h0inv)
        p[~free] = 0.0
        if bounds is not None:
            p[at_lo | at_hi] = 0.0
        pn = float(onp.linalg.norm(p))
        if pn < 1e-30:
            break
        x_new, r_new, ok = _line_search(x, p, rx, r_fn)
        if bounds is not None:
            lo, hi = bounds
            x_new = onp.clip(x_new, lo, hi)
            r_new = r_fn(x_new)
        s = x_new - x
        y = r_new - rx
        if onp.dot(s, y) > 1e-16:
            s_hist.append(s)
            y_hist.append(y)
            if len(s_hist) > m:
                s_hist.pop(0)
                y_hist.pop(0)
        x, rx = x_new, r_new
        if not ok:
            break
    return x, rx, it + 1, float(onp.linalg.norm(rx))


def solve_mech_lbfgs(case, problem_u, problem_d, sol_u, sol_d, d_irr, history_old, sol_T=None, box=False):
    sol_u = _assign_bc_sol(problem_u, sol_u)
    u0 = onp.array(sol_u[0])
    d0 = onp.array(sol_d[0])
    nu = u0.size
    free_u = _dirichlet_free_mask(problem_u)
    free_d = onp.ones(d0.size, dtype=bool)
    free = onp.concatenate([free_u, free_d])
    x0 = onp.concatenate([u0.ravel(), d0.ravel()])
    lo = onp.full_like(x0, -onp.inf)
    hi = onp.full_like(x0, onp.inf)
    lo[nu:] = onp.array(d_irr).ravel()
    hi[nu:] = 1.0
    lo[~free] = x0[~free]
    hi[~free] = x0[~free]
    bounds = (lo, hi)

    def _apply_state(x):
        u = x[:nu].reshape(u0.shape)
        d_raw = x[nu:].reshape(d0.shape)
        d = onp.maximum(onp.clip(d_raw, 0.0, 1.0), onp.array(d_irr)) if box else d_raw
        su, sd = [u], [d]
        su = _assign_bc_sol(problem_u, su)
        if sol_T is None:
            problem_u.set_params([sd[0], None])
            drive = problem_u.compute_psi_plus(su[0]) if box else problem_u.compute_history(su[0], history_old)
            problem_d.set_params([drive])
        else:
            problem_u.set_params([sd[0], sol_T[0], None])
            drive = (
                problem_u.compute_psi_plus(su[0], sol_T[0])
                if box
                else problem_u.compute_history(su[0], history_old, sol_T[0])
            )
            if case.mat.b1 != 0.0 or case.mat.b2 != 0.0:
                problem_d.set_params([drive, sol_T[0]])
            else:
                problem_d.set_params([drive])
        return su, sd

    def raw_r_fn(x):
        su, sd = _apply_state(x)
        ru = _residual(problem_u, su)
        rd = _residual(problem_d, sd)
        r = onp.concatenate([onp.array(ru), onp.array(rd)])
        if not onp.all(onp.isfinite(r)):
            r = onp.nan_to_num(r, nan=1e6, posinf=1e6, neginf=-1e6)
        return r

    # Mechanics and phase-field residuals carry different physical units.
    # Fixed block scaling is essential for a meaningful merit norm and line
    # search; without it one block can decrease while the other stagnates.
    r0_raw = raw_r_fn(x0)
    su_scale = max(float(onp.linalg.norm(r0_raw[:nu])), 1.0e-12)
    sd_scale = max(float(onp.linalg.norm(r0_raw[nu:])), 1.0e-12)
    scale_vec = onp.concatenate([onp.full(nu, su_scale), onp.full(d0.size, sd_scale)])

    def r_fn(x):
        return raw_r_fn(x) / scale_vec

    def refresh_h0(x):
        su, sd = _apply_state(x)
        raw_h0 = _make_h0inv([(problem_u, su), (problem_d, sd)])
        return lambda q: raw_h0(onp.asarray(q) * scale_vec)

    h0inv = refresh_h0(x0)
    x, rx, niter, nrm = lbfgs_minimize(
        r_fn,
        x0,
        free,
        m=20,
        maxiter=getattr(case, "lbfgs_maxiter", 40),
        tol=case.inner_tol,
        bounds=bounds if box else None,
        h0inv=h0inv,
        refresh_h0=refresh_h0,
        refresh_every=8,
    )
    u = x[:nu].reshape(u0.shape)
    d_raw = x[nu:].reshape(d0.shape)
    d = onp.maximum(onp.clip(d_raw, 0.0, 1.0), onp.array(d_irr)) if box else d_raw
    sol_u = [onp.array(_assign_bc_sol(problem_u, [u])[0])]
    sol_d = [onp.array(d)]
    if sol_T is None:
        history = onp.array(problem_u.compute_history(sol_u[0], history_old))
    else:
        history = onp.array(problem_u.compute_history(sol_u[0], history_old, sol_T[0]))
    if box:
        history = onp.array(history_old)
    return sol_u, sol_d, history, float(nrm), int(niter)


def solve_quench_lbfgs(
    case, problem_T, problem_u, problem_d, sol_u, sol_d, sol_T, sol_T_old, d_irr, history_old, dt, box=False
):
    sol_u = _assign_bc_sol(problem_u, sol_u)
    sol_T = _assign_bc_sol(problem_T, sol_T)
    u0 = onp.array(sol_u[0])
    d0 = onp.array(sol_d[0])
    T0 = onp.array(sol_T[0])
    nu, nd, nT = u0.size, d0.size, T0.size
    free = onp.concatenate(
        [_dirichlet_free_mask(problem_u), onp.ones(nd, dtype=bool), _dirichlet_free_mask(problem_T)]
    )
    x0 = onp.concatenate([u0.ravel(), d0.ravel(), T0.ravel()])
    lo = onp.full_like(x0, -onp.inf)
    hi = onp.full_like(x0, onp.inf)
    lo[nu : nu + nd] = onp.array(d_irr).ravel()
    hi[nu : nu + nd] = 1.0
    lo[~free] = x0[~free]
    hi[~free] = x0[~free]
    bounds = (lo, hi)

    def _apply_state(x):
        u = x[:nu].reshape(u0.shape)
        d_raw = x[nu : nu + nd].reshape(d0.shape)
        d = onp.maximum(onp.clip(d_raw, 0.0, 1.0), onp.array(d_irr)) if box else d_raw
        T = x[nu + nd :].reshape(T0.shape)
        su, sd, sT = [u], [d], [T]
        su = _assign_bc_sol(problem_u, su)
        sT = _assign_bc_sol(problem_T, sT)
        problem_T.set_params([sol_T_old, dt, sd[0] if case.mat.degrade_conductivity else None])
        problem_u.set_params([sd[0], sT[0], None])
        drive = (
            problem_u.compute_psi_plus(su[0], sT[0])
            if box
            else problem_u.compute_history(su[0], history_old, sT[0])
        )
        problem_d.set_params([drive, sT[0]])
        return su, sd, sT

    def raw_r_fn(x):
        su, sd, sT = _apply_state(x)
        rT = _residual(problem_T, sT)
        ru = _residual(problem_u, su)
        rd = _residual(problem_d, sd)
        r = onp.concatenate([onp.array(ru), onp.array(rd), onp.array(rT)])
        if not onp.all(onp.isfinite(r)):
            r = onp.nan_to_num(r, nan=1e6, posinf=1e6, neginf=-1e6)
        return r


    r0_raw = raw_r_fn(x0)
    cuts = (nu, nu + nd)
    su_scale = max(float(onp.linalg.norm(r0_raw[: cuts[0]])), 1.0e-12)
    sd_scale = max(float(onp.linalg.norm(r0_raw[cuts[0] : cuts[1]])), 1.0e-12)
    sT_scale = max(float(onp.linalg.norm(r0_raw[cuts[1] :])), 1.0e-12)
    scale_vec = onp.concatenate(
        [onp.full(nu, su_scale), onp.full(nd, sd_scale), onp.full(nT, sT_scale)]
    )

    def r_fn(x):
        return raw_r_fn(x) / scale_vec

    def refresh_h0(x):
        su, sd, sT = _apply_state(x)
        raw_h0 = _make_h0inv([(problem_u, su), (problem_d, sd), (problem_T, sT)])
        return lambda q: raw_h0(onp.asarray(q) * scale_vec)

    h0inv = refresh_h0(x0)
    x, rx, niter, nrm = lbfgs_minimize(
        r_fn,
        x0,
        free,
        m=20,
        maxiter=getattr(case, "lbfgs_maxiter", 30),
        tol=case.inner_tol,
        bounds=bounds if box else None,
        h0inv=h0inv,
        refresh_h0=refresh_h0,
        refresh_every=8,
    )
    u = x[:nu].reshape(u0.shape)
    d_raw = x[nu : nu + nd].reshape(d0.shape)
    d = onp.maximum(onp.clip(d_raw, 0.0, 1.0), onp.array(d_irr)) if box else d_raw
    T = x[nu + nd :].reshape(T0.shape)
    sol_u = [onp.array(_assign_bc_sol(problem_u, [u])[0])]
    sol_d = [onp.array(d)]
    sol_T = [onp.array(_assign_bc_sol(problem_T, [T])[0])]
    history = onp.array(problem_u.compute_history(sol_u[0], history_old, sol_T[0]))
    if box:
        history = onp.array(history_old)
    return sol_u, sol_d, sol_T, history, float(nrm), int(niter)
