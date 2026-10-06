"""Staggered solvers and case driver."""

from __future__ import annotations

import json
import importlib
import logging
import os
import time
from pathlib import Path

import jax_tpff.jax_compat  # noqa: F401
import jax
import jax.numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as onp
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import pyamg

from jax_fem.generate_mesh import Mesh, rectangle_mesh, box_mesh
from jax_fem.solver import solver
from jax_fem.utils import save_sol

from jax_tpff.cases import CaseSpec
from jax_tpff.problems import Elasticity, Heat, PhaseField
from jax_tpff import amr as amr_mod
from jax_tpff.lbfgs import solve_mech_lbfgs, solve_quench_lbfgs

import warnings

warnings.filterwarnings("ignore", message="VTU requires 3D points")
logging.getLogger("jax_fem").setLevel(logging.WARNING)

jax.config.update("jax_enable_x64", True)

_jax_fem_solver_module = importlib.import_module("jax_fem.solver")


def _pyamg_solve(A, b, vec, points=None):
    """AMG-preconditioned CG for the symmetric FE increment equations."""
    ai, aj, av = A.getValuesCSR()
    matrix = sp.csr_matrix((av, aj, ai))
    row_nnz = onp.diff(matrix.indptr)
    singleton = onp.flatnonzero(row_nnz == 1)
    constrained = onp.zeros(matrix.shape[0], dtype=bool)
    if singleton.size:
        starts = matrix.indptr[singleton]
        is_identity = (
            matrix.indices[starts] == singleton
        ) & onp.isclose(matrix.data[starts], 1.0)
        constrained[singleton[is_identity]] = True
    free = (~constrained).astype(float)
    projector = sp.diags(free, format="csr")
    matrix = (projector @ matrix @ projector + sp.diags(constrained.astype(float))).tocsr()
    n = matrix.shape[0]
    # Symmetric diagonal equilibration preserves the SPD system while
    # reducing the stiffness contrast seen by the multigrid hierarchy.
    sqrt_diag = onp.sqrt(onp.maximum(onp.abs(matrix.diagonal()), 1.0e-30))
    inv_sqrt_diag = 1.0 / sqrt_diag
    scaling = sp.diags(inv_sqrt_diag, format="csr")
    scaled_matrix = (scaling @ matrix @ scaling).tocsr()
    if vec > 1 and n % vec == 0:
        n_rigid = vec
        coords = None if points is None else onp.asarray(points, dtype=float)
        if vec == 2 and coords is not None and len(coords) * vec == n:
            n_rigid += 1
        near_null = onp.zeros((n, n_rigid), dtype=float)
        for component in range(vec):
            near_null[component::vec, component] = 1.0
        if n_rigid == 3:
            centred = coords[:, :2] - onp.mean(coords[:, :2], axis=0)
            scale = max(float(onp.max(onp.linalg.norm(centred, axis=1))), 1.0)
            near_null[0::2, 2] = -centred[:, 1] / scale
            near_null[1::2, 2] = centred[:, 0] / scale
    else:
        near_null = onp.ones((n, 1), dtype=float)
    # If x is a physical near-null mode and x=S y, then y=S^{-1}x.
    near_null *= sqrt_diag[:, None]
    hierarchy = pyamg.smoothed_aggregation_solver(
        scaled_matrix,
        B=near_null,
        symmetry="symmetric",
        max_coarse=40,
    )
    rhs = onp.asarray(b, dtype=float)
    y = hierarchy.solve(
        inv_sqrt_diag * rhs,
        tol=1.0e-6,
        maxiter=180,
        accel="cg",
    )
    x = inv_sqrt_diag * y
    relative = onp.linalg.norm(matrix @ x - rhs) / max(onp.linalg.norm(rhs), 1.0e-30)
    if not onp.isfinite(relative) or relative > 1.0e-5:
        raise RuntimeError(f"PyAMG-CG failed: relative residual {relative:.3e}")
    return x

def _zero(_p):
    return 0.0


def _make_load(disp):
    def val_fn(_p):
        return disp
    return val_fn


def _ele_type(case: CaseSpec) -> str:
    if case.amr:
        return "TRI3"
    if case.mode == "quench3d" or case.ele_type == "HEX8":
        return "HEX8"
    return case.ele_type or "QUAD4"


def _gauss_order(case: CaseSpec):
    et = _ele_type(case)
    if et == "TRI3":
        return 2
    return None


def _dim(case: CaseSpec) -> int:
    return 3 if case.mode == "quench3d" else 2


def _hmin(mesh: Mesh) -> float:
    return float(onp.min(amr_mod.h_size(onp.array(mesh.points[:, :2]), onp.array(mesh.cells))))


def _compact(points, cells):
    used = onp.unique(cells.reshape(-1))
    remap = -onp.ones(len(points), dtype=onp.int32)
    remap[used] = onp.arange(len(used), dtype=onp.int32)
    return onp.asarray(points)[used], remap[cells]


def _quad_mesh(case: CaseSpec) -> Mesh:
    meshio_mesh = rectangle_mesh(Nx=case.Nx, Ny=case.Ny, domain_x=case.Lx, domain_y=case.Ly)
    points = onp.asarray(meshio_mesh.points)
    cells = onp.asarray(meshio_mesh.cells_dict["quad"])
    if case.geometry == "rect":
        return Mesh(points, cells, ele_type="QUAD4")
    centers = points[cells][:, :, :2].mean(axis=1)
    if case.geometry == "lshape":
        drop = (centers[:, 0] > 250.0) & (centers[:, 1] > 250.0)
    elif case.geometry == "hole":
        drop = (centers[:, 0] - case.hole_cx) ** 2 + (centers[:, 1] - case.hole_cy) ** 2 < case.hole_r ** 2
    else:
        raise ValueError(case.geometry)
    points, cells = _compact(points, cells[~drop])
    return Mesh(points, cells, ele_type="QUAD4")


def _mesh(case: CaseSpec) -> Mesh:
    et = _ele_type(case)
    if case.geometry != "rect":
        return _quad_mesh(case)
    if et == "TRI3":
        points, cells = amr_mod.grid_to_tris(case.Nx, case.Ny, case.Lx, case.Ly)
        return Mesh(points, cells, ele_type="TRI3")
    if et == "HEX8":
        meshio_mesh = box_mesh(
            Nx=case.Nx, Ny=case.Ny, Nz=case.Nz, domain_x=case.Lx, domain_y=case.Ly, domain_z=case.Lz
        )
        return Mesh(meshio_mesh.points, meshio_mesh.cells_dict["hexahedron"], ele_type="HEX8")
    return _quad_mesh(case)


def _boundaries(case: CaseSpec):
    Lx, Ly = case.Lx, case.Ly

    def y_min(p):
        return np.isclose(p[1], 0.0, atol=1e-8)

    def y_max(p):
        return np.isclose(p[1], Ly, atol=1e-8)

    def x_min(p):
        return np.isclose(p[0], 0.0, atol=1e-8)

    def x_max(p):
        return np.isclose(p[0], Lx, atol=1e-8)

    return y_min, y_max, x_min, x_max


def _cold_mask(pts, case: CaseSpec):
    if _dim(case) == 3:
        return onp.abs(pts[:, 0] - 0.0) < 1e-8
    return (onp.abs(pts[:, 0] - 0.0) < 1e-8) | (onp.abs(pts[:, 1] - 0.0) < 1e-8) | (
        onp.abs(pts[:, 1] - case.Ly) < 1e-8
    )


def _wall_T(case: CaseSpec, t: float):
    if getattr(case, "T_reheat", None) is not None and t >= case.T_reheat:
        return case.mat.T_ref
    return case.T_cold


def _apply_wall_T(sol_T, pts, case: CaseSpec, Tval):
    if case.thermal_bc != "dirichlet":
        return sol_T
    T = onp.array(sol_T[0])
    T[_cold_mask(pts, case)] = Tval
    return [T]


def _apply_notch(points, sol_d, case: CaseSpec):
    if not case.notch:
        return sol_d
    pts = onp.asarray(points)
    x = pts[:, 0]
    y = pts[:, 1]
    sol_d = onp.array(sol_d)
    style = case.notch_style
    if style == "tpb":
        xc = 0.5 * case.Lx
        height = 0.4
        half_base = 0.1
        half = half_base * (1.0 - y / height)
        inside = (y >= -1e-12) & (y <= height) & (onp.abs(x - xc) <= onp.maximum(half, 0.0) + 1e-9)
        tip = (onp.abs(x - xc) <= 0.25 * case.mat.ell) & (onp.abs(y - height) <= 0.25 * case.mat.ell)
        sol_d[inside | tip] = 1.0
        return sol_d
    if style == "left":
        y0 = 0.5 * case.Ly if case.notch_y < 0.0 else case.notch_y
        hy = 0.25 * case.mat.ell
        sol_d[(x <= case.notch_length + 1e-12) & (onp.abs(y - y0) <= hy)] = 1.0
        return sol_d
    hy = 0.25 * case.mat.ell
    flag = (onp.abs(y - 0.5 * case.Ly) <= hy) & (x >= 0.5 * case.Lx - 1e-12)
    sol_d[flag] = 1.0
    return sol_d


def _mech_dirichlet(case: CaseSpec, disp: float):
    y_min, y_max, _x_min, _x_max = _boundaries(case)
    if case.mode == "patch":
        strain = disp / case.Ly

        def outer(p):
            return (
                np.isclose(p[0], 0.0, atol=1e-8)
                | np.isclose(p[0], case.Lx, atol=1e-8)
                | np.isclose(p[1], 0.0, atol=1e-8)
                | np.isclose(p[1], case.Ly, atol=1e-8)
            )

        def affine_x(p):
            return -case.mat.nu * strain * p[0]

        def affine_y(p):
            return strain * p[1]

        return (
            [outer, outer],
            [0, 1],
            [affine_x, affine_y],
        ), y_max, np.array([0.0, 1.0])
    if case.mode == "tension" or case.mode == "hole":
        return (
            [y_min, y_min, y_max, y_max],
            [0, 1, 0, 1],
            [_zero, _zero, _zero, _make_load(disp)],
        ), y_max, np.array([0.0, 1.0])
    if case.mode == "shear":
        return (
            [y_min, y_min, y_max, y_max],
            [0, 1, 0, 1],
            [_zero, _zero, _make_load(disp), _zero],
        ), y_max, np.array([0.0, 1.0])
    if case.mode == "tpb":
        half = case.load_half

        def pin_left(p):
            return np.isclose(p[0], 0.0, atol=1e-8) & np.isclose(p[1], 0.0, atol=1e-8)

        def pin_right(p):
            return np.isclose(p[0], case.Lx, atol=1e-8) & np.isclose(p[1], 0.0, atol=1e-8)

        def load_patch(p):
            return np.isclose(p[1], case.Ly, atol=1e-8) & (np.abs(p[0] - 0.5 * case.Lx) <= half)

        return (
            [pin_left, pin_left, pin_right, load_patch],
            [0, 1, 1, 1],
            [_zero, _zero, _zero, _make_load(disp)],
        ), load_patch, np.array([0.0, 1.0])
    if case.mode == "lshape":
        half = case.load_half
        x_load = case.load_x
        y_load = case.load_y

        def load_patch(p):
            return np.isclose(p[1], y_load, atol=1e-6) & (np.abs(p[0] - x_load) <= half)

        return (
            [y_min, y_min, load_patch],
            [0, 1, 1],
            [_zero, _zero, _make_load(disp)],
        ), load_patch, np.array([0.0, 1.0])
    raise ValueError(case.mode)


def _build_mech_problems(case: CaseSpec, mesh: Mesh, disp: float):
    dirichlet_bc_info, loc_trac, normal = _mech_dirichlet(case, disp)
    et = _ele_type(case)
    gq = _gauss_order(case)
    dim = _dim(case)
    problem_u = Elasticity(
        mesh,
        vec=dim,
        dim=dim,
        ele_type=et,
        gauss_order=gq,
        dirichlet_bc_info=dirichlet_bc_info,
        additional_info=(case.mat,),
    )
    problem_d = PhaseField(
        mesh, vec=1, dim=dim, ele_type=et, gauss_order=gq, additional_info=(case.mat,)
    )
    return problem_u, problem_d, loc_trac, normal


def _build_quench_problems(case: CaseSpec, mesh: Mesh, T_wall=None):
    y_min, y_max, x_min, x_max = _boundaries(case)
    et = _ele_type(case)
    T_wall = case.T_cold if T_wall is None else T_wall

    def T_wall_fn(_p):
        return T_wall

    dim = _dim(case)
    thermal_surfaces = None
    if dim == 3:
        dirichlet_T = [[x_min], [0], [T_wall_fn]]
        if case.thermal_bc == "convective":
            dirichlet_T = None
            thermal_surfaces = [x_min]
        dirichlet_u = [[x_max, x_max, x_max], [0, 1, 2], [_zero, _zero, _zero]]
    else:
        dirichlet_T = [[x_min, y_min, y_max], [0, 0, 0], [T_wall_fn, T_wall_fn, T_wall_fn]]
        if case.thermal_bc == "convective":
            dirichlet_T = None
            thermal_surfaces = [x_min, y_min, y_max]
        if case.mechanical_bc == "minimal":
            def pin(p):
                return np.isclose(p[0], case.Lx, atol=1e-8) & np.isclose(p[1], 0.0, atol=1e-8)

            def roller(p):
                return np.isclose(p[0], case.Lx, atol=1e-8) & np.isclose(p[1], case.Ly, atol=1e-8)

            dirichlet_u = [[pin, pin, roller], [0, 1, 0], [_zero, _zero, _zero]]
        else:
            dirichlet_u = [[x_max, x_max], [0, 1], [_zero, _zero]]
    gq = _gauss_order(case)
    vec_u = dim
    problem_T = Heat(
        mesh, vec=1, dim=dim, ele_type=et, gauss_order=gq,
        dirichlet_bc_info=dirichlet_T, location_fns=thermal_surfaces,
        additional_info=(case.mat, case.h_conv, case.T_ambient, case.lump_heat_capacity),
    )
    problem_u = Elasticity(
        mesh, vec=vec_u, dim=dim, ele_type=et, gauss_order=gq, dirichlet_bc_info=dirichlet_u, additional_info=(case.mat,)
    )
    problem_d = PhaseField(
        mesh, vec=1, dim=dim, ele_type=et, gauss_order=gq, additional_info=(case.mat,)
    )
    return problem_T, problem_u, problem_d


def _try_adapt(case: CaseSpec, mesh: Mesh, sol_u, sol_d, history, sol_T=None):
    if not case.amr:
        return mesh, sol_u, sol_d, history, sol_T, False
    pts = onp.array(mesh.points[:, :2])
    cells = onp.array(mesh.cells)
    if len(cells) >= case.amr_max_cells:
        return mesh, sol_u, sol_d, history, sol_T, False
    T = None if sol_T is None else onp.array(sol_T[0])
    marked = amr_mod.mark_cells(
        pts,
        cells,
        onp.array(sol_d[0]),
        history,
        T,
        case.mat.ell,
        d_th=case.amr_d_th,
        h_over_ell=case.amr_h_over_ell,
        buffer=case.amr_buffer,
        notch=case.notch,
        Lx=case.Lx,
        Ly=case.Ly,
        notch_style=case.notch_style,
    )
    if not onp.any(marked):
        return mesh, sol_u, sol_d, history, sol_T, False
    new_pts, new_cells, changed = amr_mod.refine_once(pts, cells, marked)
    if not changed:
        return mesh, sol_u, sol_d, history, sol_T, False
    nq = 1 if history.ndim == 1 else history.shape[1]
    u_new = amr_mod.interpolate_nodal(pts, cells, sol_u[0], new_pts)
    d_new = amr_mod.interpolate_nodal(pts, cells, sol_d[0], new_pts)
    d_new = _apply_notch(new_pts, d_new, case)
    h_new = amr_mod.interpolate_history(pts, cells, history, new_pts, new_cells, nq)
    T_new = None
    if sol_T is not None:
        T_new = [amr_mod.interpolate_nodal(pts, cells, sol_T[0], new_pts)]
    new_mesh = Mesh(new_pts, new_cells, ele_type=_ele_type(case))
    return new_mesh, [u_new], [d_new], h_new, T_new, True


def _adapt_loop(case: CaseSpec, mesh: Mesh, sol_u, sol_d, history, sol_T, max_pass: int, reset_u: bool = True):
    adapted = False
    for _ in range(max_pass):
        if len(mesh.cells) >= case.amr_max_cells:
            break
        mesh, sol_u, sol_d, history, sol_T, changed = _try_adapt(
            case, mesh, sol_u, sol_d, history, sol_T
        )
        if not changed:
            break
        adapted = True
        print(
            f"[{case.name}] AMR  cells={len(mesh.cells)}  nodes={len(mesh.points)}  hmin={_hmin(mesh):.4e}",
            flush=True,
        )
    if adapted and reset_u:
        sol_u = [onp.zeros((len(mesh.points), _dim(case)))]
    return mesh, sol_u, sol_d, history, sol_T, adapted


def _initial_adapt(case: CaseSpec, mesh: Mesh, sol_d, sol_T=None):
    if not case.amr:
        return mesh, sol_d, sol_T
    history = onp.zeros((len(mesh.cells), 1))
    sol_u = [onp.zeros((len(mesh.points), _dim(case)))]
    mesh, sol_u, sol_d, history, sol_T, _ = _adapt_loop(
        case, mesh, sol_u, sol_d, history, sol_T, case.amr_max_level, reset_u=True
    )
    return mesh, sol_d, sol_T


def _init_notch(mesh: Mesh, case: CaseSpec):
    sol_d = onp.zeros((len(mesh.points), 1))
    return _apply_notch(mesh.points, sol_d, case)


def _project_d(sol_d, d_irr):
    return [onp.clip(onp.maximum(onp.array(sol_d[0]), d_irr), 0.0, 1.0)]


def _phase_active_set(problem, guess, lower, tol=1.0e-10, maxiter=80):
    """Solve the convex AT1/AT2 damage subproblem with bound constraints.

    For fixed history and temperature these subproblems are quadratic.  The
    primal-dual active set is formed from a diagonally scaled projected
    gradient and the reduced linear system is solved exactly on the free set.
    """
    x = onp.clip(onp.asarray(guess[0], dtype=float).reshape(-1), onp.asarray(lower).reshape(-1), 1.0)
    lb = onp.asarray(lower, dtype=float).reshape(-1)
    ub = onp.ones_like(lb)
    res_list = problem.newton_update([x.reshape((-1, 1))])
    residual = onp.asarray(res_list[0]).reshape(-1)
    n = x.size
    K = sp.csr_matrix((onp.asarray(problem.V), (onp.asarray(problem.I), onp.asarray(problem.J))), shape=(n, n))
    K.sum_duplicates()
    rhs_full = K @ x - residual
    diag = onp.maximum(onp.abs(K.diagonal()), 1.0e-30)
    previous = None
    converged = False
    iteration = 0
    for iteration in range(1, maxiter + 1):
        grad = K @ x - rhs_full
        trial = x - grad / diag
        active_lower = trial <= lb + tol
        active_upper = (~active_lower) & (trial >= ub - tol)
        free = ~(active_lower | active_upper)
        status = active_lower.astype(onp.int8) + 2 * active_upper.astype(onp.int8)
        x_new = x.copy()
        x_new[active_lower] = lb[active_lower]
        x_new[active_upper] = ub[active_upper]
        if onp.any(free):
            active = ~free
            reduced_rhs = rhs_full[free]
            if onp.any(active):
                reduced_rhs = reduced_rhs - K[free][:, active] @ x_new[active]
            x_new[free] = spla.spsolve(K[free][:, free].tocsc(), reduced_rhs)
        x_new = onp.clip(x_new, lb, ub)
        grad_new = K @ x_new - rhs_full
        projected = x_new - onp.clip(x_new - grad_new / diag, lb, ub)
        projected_inf = float(onp.max(onp.abs(projected)))
        if previous is not None and onp.array_equal(status, previous) and projected_inf <= tol:
            x = x_new
            converged = True
            break
        previous = status
        x = x_new
    grad = K @ x - rhs_full
    projected = x - onp.clip(x - grad / diag, lb, ub)
    trial = x - grad / diag
    active_lower = trial <= lb + tol
    active_upper = (~active_lower) & (trial >= ub - tol)
    free = ~(active_lower | active_upper)
    stats = {
        "iterations": int(iteration),
        "converged": bool(converged or float(onp.max(onp.abs(projected))) <= tol),
        "projected_residual_inf": float(onp.max(onp.abs(projected))),
        "active_lower": int(onp.sum(active_lower)),
        "active_upper": int(onp.sum(active_upper)),
        "free": int(onp.sum(free)),
    }
    problem.last_active_set = stats
    return [x.reshape((-1, 1))]


def _solve_damage(problem, case: CaseSpec, guess, lower):
    if case.irreversibility_solver == "active-set" and case.mat.model in ("AT1", "AT2"):
        return _phase_active_set(problem, guess, lower)
    result = _project_d(_safe_solver(problem, case, guess), lower)
    problem.last_active_set = {
        "iterations": 0,
        "converged": True,
        "projected_residual_inf": float("nan"),
        "active_lower": int(onp.sum(onp.isclose(result[0], lower))),
        "active_upper": int(onp.sum(onp.isclose(result[0], 1.0))),
        "free": int(onp.size(result[0]) - onp.sum(onp.isclose(result[0], lower)) - onp.sum(onp.isclose(result[0], 1.0))),
    }
    return result


def _match_history(history, problem):
    nc = problem.fes[0].num_cells
    nq = problem.fes[0].num_quads
    h = onp.asarray(history)
    if h.ndim == 1:
        h = h[:, None]
    if h.shape[0] != nc:
        return onp.zeros((nc, nq))
    if h.shape[1] != nq:
        h = onp.repeat(onp.max(h, axis=1, keepdims=True), nq, axis=1)
    return h


def _d_stats(sol_d):
    d = onp.array(sol_d).ravel()
    return float(onp.max(d)), float(onp.mean(d)), int(onp.sum(d > 0.8))


def _solver_options(case: CaseSpec, guess):
    if case.solver == "jax-krylov":
        linear_solver = {"jax_solver": {"precond": True}}
    elif case.solver == "petsc":
        linear_solver = {"petsc_solver": {"ksp_type": "bcgsl", "pc_type": "ilu"}}
    else:
        linear_solver = {"umfpack_solver": {}}
    return {
        **linear_solver,
        "initial_guess": guess,
        "tol": 1e-6,
        "rel_tol": 1e-5,
        "maxiter": 12,
    }


def _safe_solver(problem, case: CaseSpec, guess):
    if case.solver == "pyamg":
        vec = int(problem.fes[0].vec)
        points = onp.asarray(problem.fes[0].points)
        _jax_fem_solver_module.umfpack_solve = lambda A, b: _pyamg_solve(A, b, vec, points)
    try:
        sol = solver(problem, _solver_options(case, guess))
        if all(onp.all(onp.isfinite(onp.asarray(s))) for s in sol):
            return sol
        raise FloatingPointError("non-finite Newton iterate")
    except Exception as exc:
        print(f"[{case.name}] Newton retry with zero guess ({type(exc).__name__}: {exc})", flush=True)
        zeros = [onp.zeros_like(onp.asarray(g)) for g in guess]
        try:
            return solver(problem, _solver_options(case, zeros))
        except Exception as exc2:
            print(f"[{case.name}] Newton failed twice ({type(exc2).__name__}); keep previous iterate", flush=True)
            return guess


def _prescribed_T(mesh: Mesh, case: CaseSpec):
    if case.T_top is None:
        return None
    t_bot = case.T_bot if case.T_bot is not None else case.mat.T_ref
    y = onp.asarray(mesh.points[:, 1:2])
    return [t_bot + (case.T_top - t_bot) * (y / max(case.Ly, 1e-30))]


def _stagger_mech(case, problem_u, problem_d, sol_u, sol_d, d_irr, history_old, sol_T=None):
    if case.solver in ("lbfgs", "lbfgsb"):
        return solve_mech_lbfgs(
            case, problem_u, problem_d, sol_u, sol_d, d_irr, history_old, sol_T, box=case.solver == "lbfgsb"
        )
    sol_u_inner = onp.array(sol_u[0])
    sol_d_inner = onp.array(sol_d[0])
    err = 1.0
    inner = 0
    history = history_old
    while err > case.inner_tol and inner < case.max_inner:
        if sol_T is None:
            problem_u.set_params([sol_d[0], None])
            sol_u = _safe_solver(problem_u, case, sol_u)
            history = problem_u.compute_history(sol_u[0], history_old)
            problem_d.set_params([history])
        else:
            problem_u.set_params([sol_d[0], sol_T[0], None])
            sol_u = _safe_solver(problem_u, case, sol_u)
            history = problem_u.compute_history(sol_u[0], history_old, sol_T[0])
            if case.mat.b1 != 0.0 or case.mat.b2 != 0.0:
                problem_d.set_params([history, sol_T[0]])
            else:
                problem_d.set_params([history])
        sol_d = _solve_damage(problem_d, case, sol_d, d_irr)
        err = max(
            float(onp.linalg.norm(onp.array(sol_u[0]) - sol_u_inner)),
            float(onp.linalg.norm(onp.array(sol_d[0]) - sol_d_inner)),
        )
        sol_u_inner = onp.array(sol_u[0])
        sol_d_inner = onp.array(sol_d[0])
        inner += 1
    return sol_u, sol_d, history, err, inner


def _stagger_quench(case, problem_T, problem_u, problem_d, sol_u, sol_d, sol_T, sol_T_old, d_irr, history_old, dt):
    if case.solver in ("lbfgs", "lbfgsb"):
        return solve_quench_lbfgs(
            case,
            problem_T,
            problem_u,
            problem_d,
            sol_u,
            sol_d,
            sol_T,
            sol_T_old,
            d_irr,
            history_old,
            dt,
            box=case.solver == "lbfgsb",
        )
    sol_u_inner = onp.array(sol_u[0])
    sol_d_inner = onp.array(sol_d[0])
    sol_T_inner = onp.array(sol_T[0])
    err = 1.0
    inner = 0
    history = history_old
    # With constant conductivity the thermal subproblem is independent of
    # both displacement and damage.  Solve it once at the new time level;
    # repeating the identical linear solve inside the staggered loop changes
    # neither the fixed point nor the convergence test.
    if not case.mat.degrade_conductivity:
        problem_T.set_params([sol_T_old, dt, None])
        sol_T = _safe_solver(problem_T, case, sol_T)
    while err > case.inner_tol and inner < case.max_inner:
        if case.mat.degrade_conductivity:
            problem_T.set_params([sol_T_old, dt, sol_d[0]])
            sol_T = _safe_solver(problem_T, case, sol_T)
        problem_u.set_params([sol_d[0], sol_T[0], None])
        sol_u = _safe_solver(problem_u, case, sol_u)
        history = problem_u.compute_history(sol_u[0], history_old, sol_T[0])
        problem_d.set_params([history, sol_T[0]])
        sol_d = _solve_damage(problem_d, case, sol_d, d_irr)
        du = onp.array(sol_u[0]) - sol_u_inner
        dd = onp.array(sol_d[0]) - sol_d_inner
        dT = onp.array(sol_T[0]) - sol_T_inner
        # Dimensionless RMS increments.  Raw L2 norms mix displacement,
        # damage and temperature units and grow with mesh size.
        u_scale = max(case.mat.alpha * abs(case.mat.T_ref - case.T_cold) * max(case.Lx, case.Ly), 1.0e-12)
        T_scale = max(abs(case.mat.T_ref - case.T_cold), 1.0)
        err_u = float(onp.sqrt(onp.mean(du * du)) / u_scale)
        err_d = float(onp.sqrt(onp.mean(dd * dd)))
        err_T = float(onp.sqrt(onp.mean(dT * dT)) / T_scale)
        err = max(err_u, err_d, err_T)
        sol_u_inner = onp.array(sol_u[0])
        sol_d_inner = onp.array(sol_d[0])
        sol_T_inner = onp.array(sol_T[0])
        inner += 1
    return sol_u, sol_d, sol_T, history, err, inner


def _quench_metrics(mesh, sol_d, sol_T, case):
    """Mesh-integrated diagnostics for matched 2-D QUAD4 quench cases."""
    pts = onp.asarray(mesh.points)[:, :2]
    cells = onp.asarray(mesh.cells)
    d = onp.asarray(sol_d).reshape(-1)
    T = onp.asarray(sol_T).reshape(-1)
    if cells.shape[1] == 3:
        tri = pts[cells]
        dc = d[cells]
        Tc = T[cells]
        x1, y1 = tri[:, 0, 0], tri[:, 0, 1]
        x2, y2 = tri[:, 1, 0], tri[:, 1, 1]
        x3, y3 = tri[:, 2, 0], tri[:, 2, 1]
        det = (x2-x1)*(y3-y1) - (x3-x1)*(y2-y1)
        area = 0.5 * onp.abs(det)
        gx = (dc[:,0]*(y2-y3)+dc[:,1]*(y3-y1)+dc[:,2]*(y1-y2))/det
        gy = (dc[:,0]*(x3-x2)+dc[:,1]*(x1-x3)+dc[:,2]*(x2-x1))/det
        dbar = dc.mean(axis=1)
        om = dbar*dbar if case.mat.model == "AT2" else dbar
        gamma = (om/case.mat.ell + case.mat.ell*(gx*gx+gy*gy))/case.mat.c_omega
        Tbar = Tc.mean(axis=1)
        xi = (Tbar - case.mat.T_ref) / case.mat.T_max
        Gc = case.mat.Gc0 * (1.0 - case.mat.b1 * xi + case.mat.b2 * xi * xi)
        ctr_xy = tri.mean(axis=1)
        ctr = (ctr_xy[:,0]>=0.4*case.Lx)&(ctr_xy[:,0]<=0.8*case.Lx)&(ctr_xy[:,1]>=0.3*case.Ly)&(ctr_xy[:,1]<=0.7*case.Ly)
        return {
            "fracture_surface": float(onp.sum(gamma*area)),
            "fracture_energy": float(onp.sum(Gc*gamma*area)),
            "damage_area_d05": float(onp.sum(area[dbar>=0.5])),
            "damage_area_d08": float(onp.sum(area[dbar>=0.8])),
            "T_core_mean": float(onp.average(Tc.mean(axis=1)[ctr], weights=area[ctr])) if onp.any(ctr) else float(onp.mean(T)),
            "heat_content": float(case.mat.heat_capacity*onp.sum(Tbar*area)),
        }
    if cells.shape[1] != 4:
        return {}
    xy = pts[cells]
    dc = d[cells]
    Tc = T[cells]
    # Axis-aligned structured quads used by the production T2 campaign.
    hx = onp.maximum(xy[:, :, 0].max(axis=1) - xy[:, :, 0].min(axis=1), 1e-30)
    hy = onp.maximum(xy[:, :, 1].max(axis=1) - xy[:, :, 1].min(axis=1), 1e-30)
    area = hx * hy
    dbar = dc.mean(axis=1)
    # Bilinear-gradient estimate at the cell centre; invariant to common
    # rectangle-mesh node order through left/right and top/bottom averages.
    xmid = xy[:, :, 0].mean(axis=1)
    ymid = xy[:, :, 1].mean(axis=1)
    left = xy[:, :, 0] < xmid[:, None]
    bottom = xy[:, :, 1] < ymid[:, None]
    dl = onp.sum(dc * left, axis=1) / onp.maximum(left.sum(axis=1), 1)
    dr = onp.sum(dc * ~left, axis=1) / onp.maximum((~left).sum(axis=1), 1)
    db = onp.sum(dc * bottom, axis=1) / onp.maximum(bottom.sum(axis=1), 1)
    dtp = onp.sum(dc * ~bottom, axis=1) / onp.maximum((~bottom).sum(axis=1), 1)
    grad2 = ((dr - dl) / hx) ** 2 + ((dtp - db) / hy) ** 2
    om = dbar * dbar if case.mat.model == "AT2" else dbar
    gamma = (om / case.mat.ell + case.mat.ell * grad2) / case.mat.c_omega
    Tbar = Tc.mean(axis=1)
    xi = (Tbar - case.mat.T_ref) / case.mat.T_max
    Gc = case.mat.Gc0 * (1.0 - case.mat.b1 * xi + case.mat.b2 * xi * xi)
    ctr = (xmid >= 0.4 * case.Lx) & (xmid <= 0.8 * case.Lx) & (ymid >= 0.3 * case.Ly) & (ymid <= 0.7 * case.Ly)
    # Positive heat outflow across the three cooled boundaries.
    Tl = onp.sum(Tc * left, axis=1) / onp.maximum(left.sum(axis=1), 1)
    Tr = onp.sum(Tc * ~left, axis=1) / onp.maximum((~left).sum(axis=1), 1)
    Tb = onp.sum(Tc * bottom, axis=1) / onp.maximum(bottom.sum(axis=1), 1)
    Tt = onp.sum(Tc * ~bottom, axis=1) / onp.maximum((~bottom).sum(axis=1), 1)
    kval = case.mat.kappa0 * onp.ones_like(dbar)
    if case.mat.degrade_conductivity:
        kval *= case.mat.thermal_residual_k + (1.0-case.mat.thermal_residual_k)*(1.0-dbar)**2
    on_left = onp.isclose(xy[:,:,0].min(axis=1), 0.0)
    on_bot = onp.isclose(xy[:,:,1].min(axis=1), 0.0)
    on_top = onp.isclose(xy[:,:,1].max(axis=1), case.Ly)
    qout = onp.sum(kval[on_left]*(Tr[on_left]-Tl[on_left])/hx[on_left]*hy[on_left])
    qout += onp.sum(kval[on_bot]*(Tt[on_bot]-Tb[on_bot])/hy[on_bot]*hx[on_bot])
    qout += onp.sum(kval[on_top]*(Tb[on_top]-Tt[on_top])/hy[on_top]*hx[on_top])
    if case.thermal_bc == "convective":
        Tinf = case.T_ambient if case.T_ambient is not None else case.T_cold
        qout = onp.sum(case.h_conv*(Tl[on_left]-Tinf)*hy[on_left])
        qout += onp.sum(case.h_conv*(Tb[on_bot]-Tinf)*hx[on_bot])
        qout += onp.sum(case.h_conv*(Tt[on_top]-Tinf)*hx[on_top])
    return {
        "fracture_surface": float(onp.sum(gamma * area)),
        "fracture_energy": float(onp.sum(Gc * gamma * area)),
        "damage_area_d05": float(onp.sum(area[dbar >= 0.5])),
        "damage_area_d08": float(onp.sum(area[dbar >= 0.8])),
        "T_core_mean": float(onp.average(Tc.mean(axis=1)[ctr], weights=area[ctr])) if onp.any(ctr) else float(onp.mean(T)),
        "heat_outflow_postprocessed": float(qout),
        "heat_content": float(case.mat.heat_capacity * onp.sum(Tbar * area)),
    }


def _residual_stats(problem, sol_list):
    """Residual norms before row elimination, split into free and fixed DOFs."""
    residual_list = problem.compute_residual(sol_list)
    residual = onp.concatenate([onp.asarray(r).reshape(-1) for r in residual_list])
    constrained = onp.zeros(residual.size, dtype=bool)
    offset = 0
    for fe in problem.fes:
        for nodes, component in zip(fe.node_inds_list, fe.vec_inds_list):
            nodes = onp.asarray(nodes, dtype=int)
            components = onp.asarray(component, dtype=int)
            if components.ndim == 0:
                components = onp.full(nodes.shape, int(components), dtype=int)
            inds = offset + nodes * fe.vec + components
            constrained[inds] = True
        offset += fe.num_total_nodes * fe.vec
    free = ~constrained
    free_inf = float(onp.max(onp.abs(residual[free]))) if onp.any(free) else 0.0
    fixed_inf = float(onp.max(onp.abs(residual[constrained]))) if onp.any(constrained) else 0.0
    scale = max(fixed_inf, float(onp.max(onp.abs(residual))), 1.0e-30)
    return {
        "free_inf": free_inf,
        "fixed_inf": fixed_inf,
        "relative_free_inf": float(free_inf / scale),
        "residual": residual,
        "constrained": constrained,
    }


def _heat_reaction_rate(problem_T, mesh, sol_T, sol_T_previous, sol_d, case, dt):
    """Outward heat rate from the unreduced thermal residual.

    At a prescribed-temperature node the assembled residual is the thermal
    reaction needed to maintain that value.  Cooling gives a negative nodal
    reaction in the weak-form sign convention used by JAX-FEM, hence the
    leading minus sign below.
    """
    problem_T.set_params([
        sol_T_previous,
        dt,
        sol_d if case.mat.degrade_conductivity else None,
    ])
    stats = _residual_stats(problem_T, [sol_T])
    if case.thermal_bc != "dirichlet":
        return float("nan"), stats
    cold = _cold_mask(onp.asarray(mesh.points), case)
    residual = stats["residual"].reshape(-1)
    return float(-onp.sum(residual[cold])), stats


def _mechanical_metrics(problem_u, sol_u, sol_d, sol_T):
    """Tensile-energy and principal-stress diagnostics at quadrature points."""
    problem_u.set_params([sol_d, sol_T, None])
    psi = onp.asarray(problem_u.compute_psi_plus(sol_u, sol_T))
    stress = onp.asarray(problem_u.compute_stress_quadrature(sol_u, sol_d, sol_T))
    eigmax = onp.linalg.eigvalsh(stress)[..., -1]
    weights = onp.asarray(problem_u.fes[0].JxW)
    residual = _residual_stats(problem_u, [sol_u])
    return {
        "tensile_energy": float(onp.sum(psi * weights)),
        "psi_plus_max": float(onp.max(psi)),
        "sigma_1_max": float(onp.max(eigmax)),
        "mechanical_residual_inf": residual["free_inf"],
        "mechanical_residual_relative": residual["relative_free_inf"],
    }


def _line_excess_resistance(coord, damage, mat):
    """Two-point Gauss integration of excess normal resistance on one line."""
    order = onp.argsort(coord)
    x = onp.asarray(coord)[order]
    d = onp.asarray(damage)[order]
    value = 0.0
    gauss = (-1.0 / onp.sqrt(3.0), 1.0 / onp.sqrt(3.0))
    for i in range(len(x) - 1):
        h = x[i + 1] - x[i]
        if h <= 0.0:
            continue
        for xi in gauss:
            n0, n1 = 0.5 * (1.0 - xi), 0.5 * (1.0 + xi)
            dq = n0 * d[i] + n1 * d[i + 1]
            gk = mat.thermal_residual_k + (1.0 - mat.thermal_residual_k) * (1.0 - dq) ** 2
            value += (1.0 / (mat.kappa0 * gk) - 1.0 / mat.kappa0) * h / 2.0
    return float(value)


def _boundary_layer_resistance(mesh, sol_d, case):
    """Resistance of actual boundary-attached profiles away from corners."""
    if not case.mat.degrade_conductivity or _dim(case) != 2:
        return {
            "R_layer_mean": 0.0,
            "R_layer_median": 0.0,
            "R_layer_min": 0.0,
            "R_layer_max": 0.0,
            "Pi_R_layer": 0.0,
        }
    pts = onp.asarray(mesh.points)[:, :2]
    d = onp.asarray(sol_d).reshape(-1)
    xs = onp.unique(onp.round(pts[:, 0], 12))
    ys = onp.unique(onp.round(pts[:, 1], 12))
    if len(xs) * len(ys) != len(pts):
        return {}
    ix = {float(x): i for i, x in enumerate(xs)}
    iy = {float(y): i for i, y in enumerate(ys)}
    grid = onp.zeros((len(ys), len(xs)))
    for p, val in zip(pts, d):
        grid[iy[float(onp.round(p[1], 12))], ix[float(onp.round(p[0], 12))]] = val
    depth_x = min(5.0 * case.mat.ell, case.Lx)
    depth_y = min(5.0 * case.mat.ell, 0.5 * case.Ly)
    use_x = xs <= depth_x + 1.0e-12
    use_y_bottom = ys <= depth_y + 1.0e-12
    use_y_top = ys >= case.Ly - depth_y - 1.0e-12
    values = []
    # Central portions suppress corner overlap between the three cooled edges.
    for j, y in enumerate(ys):
        if 2.0 * case.mat.ell <= y <= case.Ly - 2.0 * case.mat.ell:
            values.append(_line_excess_resistance(xs[use_x], grid[j, use_x], case.mat))
    for i, x in enumerate(xs):
        if 2.0 * case.mat.ell <= x <= case.Lx - case.mat.ell:
            values.append(_line_excess_resistance(ys[use_y_bottom], grid[use_y_bottom, i], case.mat))
            values.append(_line_excess_resistance(case.Ly - ys[use_y_top][::-1], grid[use_y_top, i][::-1], case.mat))
    if not values:
        return {}
    values = onp.asarray(values)
    mean = float(onp.mean(values))
    return {
        "R_layer_mean": mean,
        "R_layer_median": float(onp.median(values)),
        "R_layer_min": float(onp.min(values)),
        "R_layer_max": float(onp.max(values)),
        "Pi_R_layer": float(case.mat.kappa0 * mean / (0.5 * case.Ly)),
    }


def _save_ckpt(path: Path, **arrays):
    records = arrays.pop("records")
    onp.savez_compressed(path, records_json=json.dumps(records), **arrays)


def _load_ckpt(path: Path):
    data = onp.load(path, allow_pickle=True)
    records = json.loads(data["records_json"].item())
    return data, records


def _update_mech_bc(problem_u, case: CaseSpec, disp: float):
    dirichlet_bc_info, _loc, _normal = _mech_dirichlet(case, disp)
    problem_u.fes[0].update_Dirichlet_boundary_conditions(dirichlet_bc_info)


def _ckpt_mesh_kwargs(case: CaseSpec, mesh: Mesh):
    if not case.amr:
        return {}
    return {"points": onp.array(mesh.points), "cells": onp.array(mesh.cells)}


def _restore_amr_mesh(case: CaseSpec, data):
    names = getattr(data, "files", data.keys())
    if "points" not in names or "cells" not in names:
        return None
    return Mesh(onp.array(data["points"]), onp.array(data["cells"]), ele_type=_ele_type(case))


def run_mechanical(case: CaseSpec, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    vtk_dir = out_dir / "vtk"
    vtk_dir.mkdir(exist_ok=True)

    mesh = _mesh(case)
    sol_d = [_init_notch(mesh, case)]
    mesh, sol_d, _ = _initial_adapt(case, mesh, sol_d)
    disp0 = float(case.disps[0])
    problem_u, problem_d, loc_trac, normal = _build_mech_problems(case, mesh, disp0)
    sol_T = _prescribed_T(mesh, case)

    sol_u = [onp.zeros((len(mesh.points), _dim(case)))]
    d_irr = onp.array(sol_d[0])
    history = onp.zeros((problem_u.fes[0].num_cells, problem_u.fes[0].num_quads))
    if case.mat.model == "PFCZM" and case.mat.ft > 0.0:
        # Wu threshold: d = 0 stays elastic until the tensile energy exceeds ft^2/(2E).
        history[:] = (case.mat.ft ** 2) / (2.0 * case.mat.E)
    history_old = onp.array(history)

    records = []
    t0 = time.time()
    start_step = 0
    ckpt = out_dir / "checkpoint.npz"
    if ckpt.exists() and not (out_dir / "summary.json").exists():
        data, records = _load_ckpt(ckpt)
        restored = _restore_amr_mesh(case, data) if case.amr else None
        if case.amr and restored is None:
            print(f"[{case.name}] AMR checkpoint has no mesh; restart", flush=True)
            records = []
            start_step = 0
        else:
            if restored is not None:
                mesh = restored
                problem_u, problem_d, loc_trac, normal = _build_mech_problems(case, mesh, disp0)
                sol_T = _prescribed_T(mesh, case)
            sol_u = [onp.array(data["sol_u"])]
            sol_d = [onp.array(data["sol_d"])]
            d_irr = onp.array(data["d_irr"])
            history = onp.array(data["history"])
            history_old = onp.array(data["history_old"])
            start_step = int(data["step"]) + 1
            t0 = time.time() - float(data["wall_s"])
            print(f"[{case.name}] resume from step {start_step}  wall={time.time()-t0:.1f}s", flush=True)

    for step, disp in enumerate(case.disps):
        if step < start_step:
            continue
        _update_mech_bc(problem_u, case, float(disp))
        sol_u, sol_d, history, err, inner = _stagger_mech(
            case, problem_u, problem_d, sol_u, sol_d, d_irr, history_old, sol_T
        )
        history_old = onp.array(history)
        d_irr = onp.array(sol_d[0])
        if case.amr and step > 0 and (step % case.amr_every == 0):
            mesh, sol_u, sol_d, history, _, adapted = _adapt_loop(
                case, mesh, sol_u, sol_d, history_old, None, max_pass=2, reset_u=True
            )
            if adapted:
                d_irr = onp.array(sol_d[0])
                problem_u, problem_d, loc_trac, normal = _build_mech_problems(case, mesh, float(disp))
                history = _match_history(history, problem_u)
                history_old = onp.array(history)
                sol_T = _prescribed_T(mesh, case)
                _update_mech_bc(problem_u, case, float(disp))
                sol_u, sol_d, history, err, inner = _stagger_mech(
                    case, problem_u, problem_d, sol_u, sol_d, d_irr, history_old, sol_T
                )
                history_old = onp.array(history)
                d_irr = onp.array(sol_d[0])
        traction = problem_u.compute_traction(
            loc_trac, sol_u[0], sol_d[0], sol_T[0] if sol_T is not None else None, normal=normal
        )
        force = float(traction[1] if case.mode in ("tension", "patch", "lshape", "hole") else traction[0])
        if case.mode == "tpb":
            force = float(-traction[1])
        dmax, dmean, n_cracked = _d_stats(sol_d[0])
        rec = {
            "step": step,
            "disp": float(disp),
            "force": force,
            "force_x": float(traction[0]),
            "force_y": float(traction[1]),
            "d_max": dmax,
            "d_mean": dmean,
            "n_cracked": n_cracked,
            "n_cell": int(len(mesh.cells)),
            "n_node": int(len(mesh.points)),
            "h_min": _hmin(mesh) if case.amr else case.Lx / case.Nx,
            "inner": inner,
            "err": err,
            "wall_s": time.time() - t0,
        }
        records.append(rec)
        print(
            f"[{case.name}] step {step:03d}/{len(case.disps)-1}  disp={disp:.4e}  "
            f"F={force:.4e}  dmax={dmax:.3f}  ncr={n_cracked}  cells={rec['n_cell']}  "
            f"inner={inner}  t={rec['wall_s']:.1f}s",
            flush=True,
        )
        if step % case.save_every == 0 or step == len(case.disps) - 1:
            infos = [("u", sol_u[0]), ("d", sol_d[0])]
            if sol_T is not None:
                infos.append(("T", sol_T[0]))
            save_sol(
                problem_d.fes[0],
                sol_d[0],
                str(vtk_dir / f"step_{step:04d}.vtu"),
                point_infos=infos,
                cell_infos=[("history", onp.mean(onp.array(history), axis=1))],
            )
            if case.amr:
                amr_mod.plot_mesh(out_dir / f"mesh_{step:04d}.png", mesh.points[:, :2], mesh.cells, sol_d[0])
        _write_csv(out_dir / "history.csv", records)
        _save_ckpt(
            ckpt,
            sol_u=onp.array(sol_u[0]),
            sol_d=onp.array(sol_d[0]),
            d_irr=d_irr,
            history=onp.array(history),
            history_old=history_old,
            step=onp.array(step),
            wall_s=onp.array(time.time() - t0),
            records=records,
            **_ckpt_mesh_kwargs(case, mesh),
        )

    _plot_force(out_dir / "force_disp.png", records, xlabel="Displacement [mm]", ylabel="Reaction [kN]")
    _write_summary(out_dir, case, records, t0)
    return records


def run_quench(case: CaseSpec, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    vtk_dir = out_dir / "vtk"
    vtk_dir.mkdir(exist_ok=True)

    mesh = _mesh(case)
    T_ref = case.mat.T_ref
    T_cold = case.T_cold
    sol_d = [_init_notch(mesh, case)]
    sol_T = [T_ref * onp.ones((len(mesh.points), 1))]
    sol_T = _apply_wall_T(sol_T, mesh.points, case, T_cold)
    mesh, sol_d, sol_T = _initial_adapt(case, mesh, sol_d, sol_T)
    problem_T, problem_u, problem_d = _build_quench_problems(case, mesh)
    if sol_T is None:
        sol_T = [T_ref * onp.ones((len(mesh.points), 1))]
    sol_T = _apply_wall_T(sol_T, mesh.points, case, T_cold)
    sol_T_old = onp.array(sol_T[0])
    sol_u = [onp.zeros((len(mesh.points), _dim(case)))]
    d_irr = onp.array(sol_d[0])
    history = onp.zeros((problem_u.fes[0].num_cells, problem_u.fes[0].num_quads))
    history_old = onp.array(history)
    initial_metrics = _quench_metrics(mesh, sol_d[0], sol_T[0], case)
    heat_content_initial = initial_metrics.get("heat_content", 0.0)
    heat_out_cumulative = 0.0

    records = []
    t0 = time.time()
    times = case.times
    start_step = 1
    ckpt = out_dir / "checkpoint.npz"
    if ckpt.exists() and not (out_dir / "summary.json").exists():
        data, records = _load_ckpt(ckpt)
        restored = _restore_amr_mesh(case, data) if case.amr else None
        if case.amr and restored is None:
            print(f"[{case.name}] AMR checkpoint has no mesh; restart", flush=True)
            records = []
            start_step = 1
        else:
            if restored is not None:
                mesh = restored
                problem_T, problem_u, problem_d = _build_quench_problems(case, mesh)
            sol_u = [onp.array(data["sol_u"])]
            sol_d = [onp.array(data["sol_d"])]
            sol_T = [onp.array(data["sol_T"])]
            sol_T_old = onp.array(data["sol_T_old"])
            d_irr = onp.array(data["d_irr"])
            history = onp.array(data["history"])
            history_old = onp.array(data["history_old"])
            start_step = int(data["step"]) + 1
            if records:
                heat_content_initial = records[0].get("heat_content_initial", heat_content_initial)
                heat_out_cumulative = records[-1].get("heat_out_cumulative", 0.0)
            t0 = time.time() - float(data["wall_s"])
            print(f"[{case.name}] resume from step {start_step}  wall={time.time()-t0:.1f}s", flush=True)

    for step in range(start_step, len(times)):
        dt = float(times[step] - times[step - 1])
        d_previous_step = onp.array(d_irr)
        sol_T_previous_step = onp.array(sol_T_old)
        Tw = _wall_T(case, float(times[step]))
        if Tw != T_cold:
            problem_T, problem_u, problem_d = _build_quench_problems(case, mesh, T_wall=Tw)
            sol_T = _apply_wall_T(sol_T, mesh.points, case, Tw)
            T_cold = Tw
        sol_u, sol_d, sol_T, history, err, inner = _stagger_quench(
            case, problem_T, problem_u, problem_d, sol_u, sol_d, sol_T, sol_T_old, d_irr, history_old, dt
        )
        history_old = onp.array(history)
        d_irr = onp.array(sol_d[0])
        if case.amr and step > 1 and (step % case.amr_every == 0):
            mesh, sol_u, sol_d, history, sol_T, adapted = _adapt_loop(
                case, mesh, sol_u, sol_d, history_old, sol_T, max_pass=2, reset_u=True
            )
            if adapted:
                d_irr = onp.array(sol_d[0])
                sol_T = _apply_wall_T(sol_T, mesh.points, case, T_cold)
                sol_T_old = onp.array(sol_T[0])
                problem_T, problem_u, problem_d = _build_quench_problems(case, mesh, T_wall=T_cold)
                history = _match_history(history, problem_u)
                history_old = onp.array(history)
                # The current time level has already been solved.  Re-solving
                # the heat equation with the same dt after remeshing advances
                # it twice and produces artificial over-cooling.  The
                # interpolated current state becomes the old state for the
                # next physical increment instead.
                err = max(err, 0.0)
        dmax, dmean, n_cracked = _d_stats(sol_d[0])
        physical = _quench_metrics(mesh, sol_d[0], sol_T[0], case)
        reaction_rate, heat_residual = _heat_reaction_rate(
            problem_T,
            mesh,
            sol_T[0],
            sol_T_previous_step,
            sol_d[0],
            case,
            dt,
        )
        if case.thermal_bc == "dirichlet":
            heat_rate = reaction_rate
        else:
            heat_rate = float(physical.get("heat_outflow_postprocessed", 0.0))
        physical["heat_outflow"] = float(heat_rate)
        physical["heat_outflow_reaction"] = float(reaction_rate)
        physical["thermal_residual_inf"] = float(heat_residual["free_inf"])
        physical["thermal_residual_relative"] = float(heat_residual["relative_free_inf"])
        physical.update(_mechanical_metrics(problem_u, sol_u[0], sol_d[0], sol_T[0]))
        physical.update(_boundary_layer_resistance(mesh, sol_d[0], case))
        active = getattr(problem_d, "last_active_set", {})
        heat_out_cumulative += heat_rate * dt
        heat_balance = (
            float(physical.get("heat_content", heat_content_initial))
            - heat_content_initial
            + heat_out_cumulative
        )
        heat_scale = max(
            abs(float(physical.get("heat_content", heat_content_initial)) - heat_content_initial),
            abs(heat_out_cumulative),
            1.0e-14,
        )
        d_current_step = onp.asarray(sol_d[0])
        min_damage_increment = (
            float(onp.min(d_current_step - d_previous_step))
            if d_current_step.shape == d_previous_step.shape
            else 0.0
        )
        rec = {
            "step": step,
            "time": float(times[step]),
            "dt": dt,
            "d_max": dmax,
            "d_mean": dmean,
            "T_min": float(onp.min(sol_T[0])),
            "T_max": float(onp.max(sol_T[0])),
            "n_cell": int(len(mesh.cells)),
            "n_node": int(len(mesh.points)),
            "h_min": _hmin(mesh) if case.amr else case.Lx / case.Nx,
            "inner": inner,
            "err": err,
            "wall_s": time.time() - t0,
            "n_cracked": n_cracked,
            "converged": bool(err <= case.inner_tol),
            "min_damage_increment": min_damage_increment,
            "damage_kkt_residual": float(active.get("projected_residual_inf", float("nan"))),
            "damage_active_lower": int(active.get("active_lower", 0)),
            "damage_active_upper": int(active.get("active_upper", 0)),
            "damage_active_set_iterations": int(active.get("iterations", 0)),
            "damage_active_set_converged": bool(active.get("converged", True)),
            "heat_content_initial": float(heat_content_initial),
            "heat_out_cumulative": float(heat_out_cumulative),
            "heat_balance_residual": float(heat_balance),
            "heat_balance_relative": float(abs(heat_balance) / heat_scale),
            **physical,
        }
        sol_T_old = onp.array(sol_T[0])
        records.append(rec)
        print(
            f"[{case.name}] step {step:03d}/{len(times)-1}  t={times[step]:.4f}s  "
            f"dmax={dmax:.3f}  ncr={n_cracked}  cells={rec.get('n_cell','?')}  "
            f"T=[{rec['T_min']:.1f},{rec['T_max']:.1f}]  inner={inner}  wall={rec['wall_s']:.1f}s",
            flush=True,
        )
        if step % case.save_every == 0 or step == len(times) - 1:
            save_sol(
                problem_d.fes[0],
                sol_d[0],
                str(vtk_dir / f"step_{step:04d}.vtu"),
                point_infos=[("u", sol_u[0]), ("d", sol_d[0]), ("T", sol_T[0])],
                cell_infos=[("history", onp.mean(onp.array(history), axis=1))],
            )
            if case.amr:
                amr_mod.plot_mesh(out_dir / f"mesh_{step:04d}.png", mesh.points[:, :2], mesh.cells, sol_d[0])
        _write_csv(out_dir / "history.csv", records)
        _save_ckpt(
            ckpt,
            sol_u=onp.array(sol_u[0]),
            sol_d=onp.array(sol_d[0]),
            sol_T=onp.array(sol_T[0]),
            sol_T_old=sol_T_old,
            d_irr=d_irr,
            history=onp.array(history),
            history_old=history_old,
            step=onp.array(step),
            wall_s=onp.array(time.time() - t0),
            records=records,
            **_ckpt_mesh_kwargs(case, mesh),
        )

    _plot_force(out_dir / "dmax_time.png", records, xkey="time", ykey="d_max", xlabel="Time [s]", ylabel="max d")
    _write_summary(out_dir, case, records, t0)
    return records


def _write_csv(path: Path, records):
    if not records:
        return
    keys = list(records[0].keys())
    with open(path, "w") as f:
        f.write(",".join(keys) + "\n")
        for r in records:
            f.write(",".join(str(r[k]) for k in keys) + "\n")


def _plot_force(path: Path, records, xlabel="x", ylabel="y", xkey=None, ykey=None):
    if not records:
        return
    if xkey is None:
        x = [r.get("disp", r.get("time", 0.0)) for r in records]
        y = [r.get("force", r.get("d_max", 0.0)) for r in records]
    else:
        x = [r[xkey] for r in records]
        y = [r[ykey] for r in records]
    fig, ax = plt.subplots(figsize=(6, 4.5))
    ax.plot(x, y, "-o", ms=3)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def _write_summary(out_dir: Path, case: CaseSpec, records, t0):
    forces = [r["force"] for r in records if "force" in r]
    n_cells = [r["n_cell"] for r in records if "n_cell" in r]
    hmins = [r["h_min"] for r in records if r.get("h_min") is not None]
    hmin = float(min(hmins)) if hmins else case.Lx / case.Nx
    last = records[-1] if records else {}
    n_node = last.get("n_node")
    dim = _dim(case)
    ncomp = dim + 1
    if case.mode in ("quench", "quench3d") or case.T_top is not None:
        ncomp += 1
    summary = {
        "case": case.name,
        "mode": case.mode,
        "Nx": case.Nx,
        "Ny": case.Ny,
        "Nz": case.Nz,
        "ell": case.mat.ell,
        "Gc0": case.mat.Gc0,
        "model": case.mat.model,
        "plane_stress": case.mat.plane_stress,
        "lambda_effective_GPa": case.mat.lmbda,
        "mu_GPa": case.mat.mu,
        "temperature_dependent_Gc": bool(case.mat.b1 != 0.0 or case.mat.b2 != 0.0),
        "degrade_conductivity": case.mat.degrade_conductivity,
        "thermal_residual_k": case.mat.thermal_residual_k,
        "irreversibility_solver": case.irreversibility_solver,
        "mechanical_bc": case.mechanical_bc,
        "solver": case.solver,
        "amr": case.amr,
        "n_steps": len(records),
        "wall_s": time.time() - t0,
        "peak_force": float(max(forces)) if forces else None,
        "n_cell_init": int(n_cells[0]) if n_cells else None,
        "n_cell_final": int(n_cells[-1]) if n_cells else None,
        "n_cell_max": int(max(n_cells)) if n_cells else None,
        "h_min": hmin,
        "ell_over_hmin": float(case.mat.ell / hmin) if hmin > 0 else None,
        "n_node": n_node,
        "n_dof": int(n_node) * ncomp if n_node is not None else None,
        "n_cracked": last.get("n_cracked"),
        "T_max": last.get("T_max"),
        "last": last,
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))


def run_case(case: CaseSpec, out_root: Path):
    out_dir = out_root / case.name
    if case.mode in ("tension", "shear", "patch", "tpb", "lshape", "hole"):
        return run_mechanical(case, out_dir)
    if case.mode in ("quench", "quench3d"):
        return run_quench(case, out_dir)
    if case.mode == "inverse":
        from jax_tpff.inverse import run_inverse

        return run_inverse(case, out_dir)
    raise ValueError(case.mode)
