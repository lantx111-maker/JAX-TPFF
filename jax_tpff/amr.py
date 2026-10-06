"""Conforming longest-edge bisection AMR for TRI3 phase-field meshes.

Deal.II hanging-node AMR is not available in JAX-FEM. This module keeps a
single element type (TRI3) and recovers conformity by Rivara propagation:
if any edge of a triangle is split, its longest edge is split as well.
History is transferred by barycentric interpolation of a cell-wise maximum,
which is conservative for Miehe irreversibility.
"""

from __future__ import annotations

import numpy as onp


def edge_key(i: int, j: int) -> tuple[int, int]:
    return (i, j) if i < j else (j, i)


def grid_to_tris(Nx: int, Ny: int, Lx: float, Ly: float):
    x = onp.linspace(0.0, Lx, Nx + 1)
    y = onp.linspace(0.0, Ly, Ny + 1)
    xv, yv = onp.meshgrid(x, y, indexing="ij")
    points = onp.stack((xv.ravel(), yv.ravel()), axis=1)
    inds = onp.arange((Nx + 1) * (Ny + 1)).reshape(Nx + 1, Ny + 1)
    n0 = inds[:-1, :-1].ravel()
    n1 = inds[1:, :-1].ravel()
    n2 = inds[1:, 1:].ravel()
    n3 = inds[:-1, 1:].ravel()
    t0 = onp.stack((n0, n1, n2), axis=1)
    t1 = onp.stack((n0, n2, n3), axis=1)
    cells = onp.vstack((t0, t1)).astype(onp.int32)
    return _orient(points, cells)


def _orient(points: onp.ndarray, cells: onp.ndarray) -> tuple[onp.ndarray, onp.ndarray]:
    p0 = points[cells[:, 0]]
    p1 = points[cells[:, 1]]
    p2 = points[cells[:, 2]]
    cross = (p1[:, 0] - p0[:, 0]) * (p2[:, 1] - p0[:, 1]) - (p1[:, 1] - p0[:, 1]) * (
        p2[:, 0] - p0[:, 0]
    )
    bad = cross < 0
    if onp.any(bad):
        cells = cells.copy()
        cells[bad] = cells[bad][:, [0, 2, 1]]
    return points, cells


def triangle_areas(points: onp.ndarray, cells: onp.ndarray) -> onp.ndarray:
    p0 = points[cells[:, 0]]
    p1 = points[cells[:, 1]]
    p2 = points[cells[:, 2]]
    return 0.5 * onp.abs(
        (p1[:, 0] - p0[:, 0]) * (p2[:, 1] - p0[:, 1]) - (p1[:, 1] - p0[:, 1]) * (p2[:, 0] - p0[:, 0])
    )


def h_size(points: onp.ndarray, cells: onp.ndarray) -> onp.ndarray:
    return onp.sqrt(2.0 * triangle_areas(points, cells))


def _longest_edge(points: onp.ndarray, cell: onp.ndarray) -> tuple[int, int]:
    v0, v1, v2 = int(cell[0]), int(cell[1]), int(cell[2])
    d01 = onp.linalg.norm(points[v0] - points[v1])
    d12 = onp.linalg.norm(points[v1] - points[v2])
    d20 = onp.linalg.norm(points[v2] - points[v0])
    if d01 >= d12 and d01 >= d20:
        return edge_key(v0, v1)
    if d12 >= d20:
        return edge_key(v1, v2)
    return edge_key(v2, v0)


def _split_edges(points: onp.ndarray, cells: onp.ndarray, marked: onp.ndarray) -> set:
    split: set[tuple[int, int]] = set()
    for i, flag in enumerate(marked):
        if flag:
            split.add(_longest_edge(points, cells[i]))
    changed = True
    while changed:
        changed = False
        for cell in cells:
            v0, v1, v2 = int(cell[0]), int(cell[1]), int(cell[2])
            edges = (edge_key(v0, v1), edge_key(v1, v2), edge_key(v2, v0))
            if any(e in split for e in edges):
                le = _longest_edge(points, cell)
                if le not in split:
                    split.add(le)
                    changed = True
    return split


def _subdivide_cell(cell, mid):
    v0, v1, v2 = int(cell[0]), int(cell[1]), int(cell[2])
    m01 = mid.get(edge_key(v0, v1))
    m12 = mid.get(edge_key(v1, v2))
    m20 = mid.get(edge_key(v2, v0))
    nmid = (m01 is not None) + (m12 is not None) + (m20 is not None)
    if nmid == 0:
        return [(v0, v1, v2)]
    if nmid == 1:
        if m01 is not None:
            return [(v0, m01, v2), (m01, v1, v2)]
        if m12 is not None:
            return [(v0, v1, m12), (v0, m12, v2)]
        return [(v0, v1, m20), (m20, v1, v2)]
    if nmid == 2:
        if m01 is not None and m12 is not None:
            return [(m01, v1, m12), (v0, m01, v2), (m01, m12, v2)]
        if m12 is not None and m20 is not None:
            return [(m12, v2, m20), (v0, v1, m12), (v0, m12, m20)]
        return [(m20, v0, m01), (m01, v1, v2), (m20, m01, v2)]
    return [
        (v0, m01, m20),
        (v1, m12, m01),
        (v2, m20, m12),
        (m01, m12, m20),
    ]


def refine_once(points: onp.ndarray, cells: onp.ndarray, marked: onp.ndarray):
    if not onp.any(marked):
        return points, cells, False
    split = _split_edges(points, cells, marked)
    if not split:
        return points, cells, False
    mid = {}
    new_pts = [points[i] for i in range(len(points))]
    for i, j in split:
        mid[(i, j)] = len(new_pts)
        new_pts.append(0.5 * (points[i] + points[j]))
    children = []
    for cell in cells:
        children.extend(_subdivide_cell(cell, mid))
    new_points = onp.asarray(new_pts, dtype=onp.float64)
    new_cells = onp.asarray(children, dtype=onp.int32)
    return _orient(new_points, new_cells) + (True,)


def dilate_marks(cells: onp.ndarray, marked: onp.ndarray, layers: int = 1) -> onp.ndarray:
    marked = onp.array(marked, dtype=bool)
    if layers <= 0:
        return marked
    edge_to_cells: dict[tuple[int, int], list[int]] = {}
    for ci, cell in enumerate(cells):
        for a, b in ((cell[0], cell[1]), (cell[1], cell[2]), (cell[2], cell[0])):
            edge_to_cells.setdefault(edge_key(int(a), int(b)), []).append(ci)
    neigh = [set() for _ in range(len(cells))]
    for ids in edge_to_cells.values():
        for i in ids:
            for j in ids:
                if i != j:
                    neigh[i].add(j)
    out = marked.copy()
    for _ in range(layers):
        extra = onp.zeros_like(out)
        for i, flag in enumerate(out):
            if flag:
                for j in neigh[i]:
                    extra[j] = True
        out = out | extra
    return out


def mark_cells(
    points: onp.ndarray,
    cells: onp.ndarray,
    sol_d: onp.ndarray,
    history: onp.ndarray | None,
    sol_T: onp.ndarray | None,
    ell: float,
    d_th: float = 0.15,
    h_over_ell: float = 0.25,
    T_jump: float = 25.0,
    buffer: int = 1,
    notch: bool = False,
    Lx: float = 1.0,
    Ly: float = 1.0,
    notch_style: str = "miehe",
) -> onp.ndarray:
    h = h_size(points, cells)
    need = h > h_over_ell * ell
    d_nodes = onp.asarray(sol_d).reshape(-1)
    d_cell = onp.max(d_nodes[cells], axis=1)
    mark = (d_cell > d_th) & need
    if history is not None and history.size:
        hmean = onp.mean(onp.asarray(history).reshape(len(cells), -1), axis=1)
        h_star = float(onp.max(hmean)) if onp.max(hmean) > 0 else 1.0
        mark = mark | ((hmean > 0.05 * h_star) & need & (h_star > 1e-12))
    if sol_T is not None:
        T = onp.asarray(sol_T).reshape(-1)
        dT = onp.max(T[cells], axis=1) - onp.min(T[cells], axis=1)
        mark = mark | ((dT > T_jump) & need)
    if notch:
        cxy = onp.mean(points[cells], axis=1)
        if notch_style == "tpb":
            band = (onp.abs(cxy[:, 0] - 0.5 * Lx) <= max(0.03, 2.0 * ell)) & (cxy[:, 1] <= 1.7)
        else:
            band = (cxy[:, 0] >= 0.48 * Lx) & (onp.abs(cxy[:, 1] - 0.5 * Ly) <= 1.25 * ell)
        mark = mark | (band & need)
    return dilate_marks(cells, mark, layers=buffer)


def refine_to_indicator(
    points: onp.ndarray,
    cells: onp.ndarray,
    marked: onp.ndarray,
    max_level: int = 4,
    max_cells: int = 20000,
):
    did = False
    for _ in range(max_level):
        if len(cells) >= max_cells:
            break
        marked = marked[: len(cells)]
        if not onp.any(marked):
            break
        points, cells, changed = refine_once(points, cells, marked)
        if not changed:
            break
        did = True
        marked = onp.zeros(len(cells), dtype=bool)
        # after a global split, stop this cycle; caller re-marks on the new mesh
        break
    return points, cells, did


def barycentric(p, a, b, c):
    v0 = b - a
    v1 = c - a
    v2 = p - a
    d00 = onp.dot(v0, v0)
    d01 = onp.dot(v0, v1)
    d11 = onp.dot(v1, v1)
    d20 = onp.dot(v2, v0)
    d21 = onp.dot(v2, v1)
    denom = d00 * d11 - d01 * d01
    if abs(denom) < 1e-30:
        return -1.0, -1.0, -1.0
    v = (d11 * d20 - d01 * d21) / denom
    w = (d00 * d21 - d01 * d20) / denom
    u = 1.0 - v - w
    return u, v, w


def interpolate_nodal(old_pts, old_cells, old_sol, new_pts):
    old_sol = onp.asarray(old_sol)
    if old_sol.ndim == 1:
        old_sol = old_sol[:, None]
    n_new = len(new_pts)
    n_comp = old_sol.shape[1]
    out = onp.zeros((n_new, n_comp))
    centroids = (old_pts[old_cells[:, 0]] + old_pts[old_cells[:, 1]] + old_pts[old_cells[:, 2]]) / 3.0
    try:
        from scipy.spatial import cKDTree

        tree = cKDTree(centroids)
        _, cand = tree.query(new_pts, k=min(12, len(old_cells)))
        cand = onp.atleast_2d(cand)
    except Exception:
        cand = None
    assigned = onp.zeros(n_new, dtype=bool)
    for i, p in enumerate(new_pts):
        ids = cand[i] if cand is not None else range(len(old_cells))
        for ci in onp.atleast_1d(ids):
            a, b, c = old_cells[int(ci)]
            u, v, w = barycentric(p, old_pts[a], old_pts[b], old_pts[c])
            if min(u, v, w) >= -1e-8:
                out[i] = u * old_sol[a] + v * old_sol[b] + w * old_sol[c]
                assigned[i] = True
                break
        if not assigned[i]:
            d2 = onp.sum((old_pts - p) ** 2, axis=1)
            out[i] = old_sol[int(onp.argmin(d2))]
    return out


def interpolate_history(old_pts, old_cells, old_H, new_pts, new_cells, n_quads: int):
    old_H = onp.asarray(old_H)
    if old_H.ndim == 1:
        hcell = old_H
    else:
        hcell = onp.max(old_H.reshape(len(old_cells), -1), axis=1)
    centroids = (old_pts[old_cells[:, 0]] + old_pts[old_cells[:, 1]] + old_pts[old_cells[:, 2]]) / 3.0
    new_cent = (new_pts[new_cells[:, 0]] + new_pts[new_cells[:, 1]] + new_pts[new_cells[:, 2]]) / 3.0
    try:
        from scipy.spatial import cKDTree

        tree = cKDTree(centroids)
        _, idx = tree.query(new_cent, k=1)
        hnew = hcell[onp.atleast_1d(idx)]
    except Exception:
        hnew = onp.zeros(len(new_cells))
        for i, p in enumerate(new_cent):
            d2 = onp.sum((centroids - p) ** 2, axis=1)
            hnew[i] = hcell[int(onp.argmin(d2))]
    return onp.repeat(hnew[:, None], n_quads, axis=1)


def plot_mesh(path, points, cells, sol_d=None):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6, 5))
    d = None if sol_d is None else onp.asarray(sol_d).ravel()
    ax.tripcolor(
        points[:, 0],
        points[:, 1],
        cells,
        facecolors=onp.mean(d[cells], axis=1) if d is not None else onp.zeros(len(cells)),
        shading="flat",
        cmap="coolwarm",
        vmin=0.0,
        vmax=1.0,
    )
    ax.triplot(points[:, 0], points[:, 1], cells, color="k", lw=0.15, alpha=0.4)
    ax.set_aspect("equal")
    ax.set_xlabel("x [mm]")
    ax.set_ylabel("y [mm]")
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)
