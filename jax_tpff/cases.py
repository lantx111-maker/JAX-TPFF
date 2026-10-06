"""Frozen numerical-example specifications for the numerical study.

CPU cases are sized to run on the local JAX CPU backend.
Production cases keep the deal.II / literature meshes and length scales
and are intended for a GPU node.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Literal

import numpy as onp

from jax_tpff.constitutive import Material


Mode = Literal["tension", "shear", "patch", "quench", "quench3d", "inverse", "tpb", "lshape", "hole"]


@dataclass
class CaseSpec:
    name: str
    mode: Mode
    mat: Material
    Lx: float
    Ly: float
    Nx: int
    Ny: int
    notch: bool
    disps: onp.ndarray
    times: onp.ndarray | None
    T_cold: float | None
    max_inner: int
    inner_tol: float
    description: str
    solver: str = "umfpack"
    save_every: int = 1
    tags: tuple = ()
    amr: bool = False
    amr_max_level: int = 4
    amr_every: int = 2
    amr_d_th: float = 0.15
    amr_h_over_ell: float = 0.25
    amr_buffer: int = 1
    amr_max_cells: int = 12000
    ele_type: str = "QUAD4"
    T_top: float | None = None
    T_bot: float | None = None
    Nz: int = 1
    Lz: float = 1.0
    T_reheat: float | None = None
    lbfgs_maxiter: int = 40
    theta_true: tuple | None = None
    geometry: str = "rect"
    notch_style: str = "miehe"
    notch_length: float = 0.0
    notch_y: float = -1.0
    hole_cx: float = 0.0
    hole_cy: float = 0.0
    hole_r: float = 0.0
    load_half: float = 0.0
    load_x: float = -1.0
    load_y: float = -1.0
    thermal_bc: str = "dirichlet"
    h_conv: float = 0.0
    T_ambient: float | None = None
    lump_heat_capacity: bool = False
    irreversibility_solver: str = "projection"
    mechanical_bc: str = "clamped"


def _steel_at2(ell: float, plane_stress: bool = True) -> Material:
    return Material(E=210.0, nu=0.3, Gc0=2.7e-3, ell=ell, plane_stress=plane_stress, model="AT2")


def _steel_at1(ell: float, plane_stress: bool = True) -> Material:
    return Material(E=210.0, nu=0.3, Gc0=2.7e-3, ell=ell, plane_stress=plane_stress, model="AT1")


def _steel_thermo(ell: float) -> Material:
    return Material(
        E=210.0,
        nu=0.3,
        Gc0=2.7e-3,
        ell=ell,
        plane_stress=True,
        model="AT2",
        alpha=1.2e-5,
        T_ref=293.15,
        T_max=1000.0,
        b1=1.8,
        b2=1.1,
    )


def _steel_pfczm(ell: float) -> Material:
    return Material(
        E=210.0,
        nu=0.3,
        Gc0=2.7e-3,
        ell=ell,
        plane_stress=True,
        model="PFCZM",
        ft=0.5,
        p=2.0,
        a2=-0.5,
        a3=0.0,
    )


def ceramic_from_lame(ell: float, gc_T: bool, degrade_k: bool, plane_stress: bool = True) -> Material:
    lmbda, mu = 213.46, 142.31
    nu = lmbda / (2.0 * (lmbda + mu))
    E = mu * (3.0 * lmbda + 2.0 * mu) / (lmbda + mu)
    return Material(
        E=E,
        nu=nu,
        Gc0=42.47e-6,
        ell=ell,
        plane_stress=plane_stress,
        model="AT2",
        heat_capacity=3.5024e-3,
        kappa0=0.031,
        alpha=7.5e-6,
        T_ref=573.15,
        T_max=1000.0,
        b1=1.8 if gc_T else 0.0,
        b2=1.1 if gc_T else 0.0,
        degrade_conductivity=degrade_k,
    )


def ceramic_manuscript(
    ell: float,
    degrade_k: bool,
    *,
    gc_T: bool = False,
    model: str = "AT2",
    thermal_residual_k: float = 1.0e-2,
) -> Material:
    """Ceramic material used by the manuscript thermal-shock study.

    The principal matched pairs use a resolvable residual conductivity and a
    constant fracture toughness.  Temperature-dependent toughness and AT1
    are retained as explicit controls rather than being folded into the
    headline comparison.
    """
    base = ceramic_from_lame(ell, gc_T, degrade_k, plane_stress=True)
    return replace(base, model=model, thermal_residual_k=thermal_residual_k)


def ceramic_pfczm(ell: float, gc_T: bool, degrade_k: bool, plane_stress: bool = True, ft: float = 0.12) -> Material:
    return replace(
        ceramic_from_lame(ell, gc_T, degrade_k, plane_stress=plane_stress),
        model="PFCZM",
        ft=ft,
        p=2.0,
        a2=-0.5,
        a3=0.0,
    )


def ceramic_at1(ell: float, gc_T: bool, degrade_k: bool, plane_stress: bool = True) -> Material:
    return replace(
        ceramic_from_lame(ell, gc_T, degrade_k, plane_stress=plane_stress),
        model="AT1",
    )


def _lame_to_Enu(lmbda: float, mu: float) -> tuple[float, float]:
    nu = lmbda / (2.0 * (lmbda + mu))
    E = mu * (3.0 * lmbda + 2.0 * mu) / (lmbda + mu)
    return E, nu


def _linspace_load(start, mid, end, n_pre, n_post):
    return onp.hstack((onp.linspace(start, mid, n_pre, endpoint=False), onp.linspace(mid, end, n_post)))


CASES: dict[str, CaseSpec] = {}


def _register(case: CaseSpec):
    CASES[case.name] = case
    return case


_register(CaseSpec(
    name="M1_tension_smoke",
    mode="tension",
    mat=_steel_at2(ell=0.08, plane_stress=True),
    Lx=1.0, Ly=1.0, Nx=20, Ny=20,
    notch=True,
    disps=onp.linspace(0.0, 6.0e-3, 13),
    times=None,
    T_cold=None,
    max_inner=1,
    inner_tol=1e-4,
    description="CPU smoke: notched tension, AT2, h/ℓ≈0.63, 12 load steps.",
    save_every=1,
    tags=("cpu", "smoke", "M1"),
))

_register(CaseSpec(
    name="M1_tension_cpu",
    mode="tension",
    mat=_steel_at2(ell=0.04, plane_stress=True),
    Lx=1.0, Ly=1.0, Nx=40, Ny=40,
    notch=True,
    disps=_linspace_load(0.0, 4.0e-3, 6.0e-3, 11, 21),
    times=None,
    T_cold=None,
    max_inner=2,
    inner_tol=1e-5,
    description="CPU standard tension. Jin/Miehe unit square, coarsened ℓ=0.04 mm.",
    save_every=2,
    tags=("cpu", "M1"),
))

_register(CaseSpec(
    name="M1_tension_ref",
    mode="tension",
    mat=_steel_at2(ell=0.015, plane_stress=True),
    Lx=1.0, Ly=1.0, Nx=80, Ny=80,
    notch=True,
    disps=_linspace_load(0.0, 4.0e-3, 7.0e-3, 9, 31),
    times=None,
    T_cold=None,
    max_inner=4,
    inner_tol=1e-6,
    description="Production tension: ℓ=0.015, h=ℓ/1.2. GPU recommended.",
    save_every=2,
    tags=("gpu", "M1", "production"),
))

_register(CaseSpec(
    name="M1_shear_smoke",
    mode="shear",
    mat=_steel_at2(ell=0.08, plane_stress=True),
    Lx=1.0, Ly=1.0, Nx=20, Ny=20,
    notch=True,
    disps=onp.linspace(0.0, 1.0e-2, 11),
    times=None,
    T_cold=None,
    max_inner=1,
    inner_tol=1e-4,
    description="CPU smoke: notched shear, AT2.",
    save_every=1,
    tags=("cpu", "smoke", "M1"),
))

_register(CaseSpec(
    name="M1_shear_cpu",
    mode="shear",
    mat=_steel_at2(ell=0.04, plane_stress=True),
    Lx=1.0, Ly=1.0, Nx=40, Ny=40,
    notch=True,
    disps=onp.linspace(0.0, 1.2e-2, 25),
    times=None,
    T_cold=None,
    max_inner=2,
    inner_tol=1e-5,
    description="CPU standard shear. Jin simple-shear deck, coarsened ℓ.",
    save_every=2,
    tags=("cpu", "M1"),
))

_register(CaseSpec(
    name="T2_quench_smoke",
    mode="quench",
    mat=ceramic_from_lame(ell=0.8, gc_T=True, degrade_k=False),
    Lx=12.5, Ly=5.0, Nx=25, Ny=10,
    notch=False,
    disps=onp.array([0.0]),
    times=onp.linspace(0.0, 0.05, 11),
    T_cold=293.15,
    max_inner=1,
    inner_tol=1e-4,
    description="CPU smoke quench: half-scale plate, ℓ=0.8 mm, 10 thermal steps.",
    save_every=1,
    tags=("cpu", "smoke", "T2"),
))

_register(CaseSpec(
    name="T2_quench_cpu",
    mode="quench",
    mat=ceramic_from_lame(ell=0.4, gc_T=True, degrade_k=False),
    Lx=12.5, Ly=5.0, Nx=40, Ny=16,
    notch=False,
    disps=onp.array([0.0]),
    times=onp.linspace(0.0, 0.10, 21),
    T_cold=293.15,
    max_inner=2,
    inner_tol=1e-5,
    description="CPU quench with temperature-dependent toughness and constant conductivity.",
    save_every=2,
    tags=("cpu", "T2"),
))

_register(CaseSpec(
    name="T2_quench_cpu_kappa",
    mode="quench",
    mat=ceramic_from_lame(ell=0.4, gc_T=True, degrade_k=True),
    Lx=12.5, Ly=5.0, Nx=40, Ny=16,
    notch=False,
    disps=onp.array([0.0]),
    times=onp.linspace(0.0, 0.10, 21),
    T_cold=293.15,
    max_inner=2,
    inner_tol=1e-5,
    description="CPU quench with Gc(T) and kappa(d) shielding.",
    save_every=2,
    tags=("cpu", "T2"),
))

# P0 matched campaign: ell/h=4 on identical 100x40 meshes.  The 11-step
# schedule keeps the fully coupled convergence study tractable on CPU while
# retaining the 0.1 s terminal state used in the manuscript.
for _name, _mat, _desc in (
    ("T2_p0_at2_k0", ceramic_from_lame(0.5, True, False), "AT2, constant conductivity"),
    ("T2_p0_at2_kd", ceramic_from_lame(0.5, True, True), "AT2, damage-dependent conductivity"),
    ("T2_p0_at1_k0", ceramic_at1(0.5, True, False), "AT1 strength-threshold surrogate, constant conductivity"),
    ("T2_p0_at1_kd", ceramic_at1(0.5, True, True), "AT1 strength-threshold surrogate, damage-dependent conductivity"),
):
    _register(CaseSpec(
        name=_name,
        mode="quench",
        mat=_mat,
        Lx=12.5, Ly=5.0, Nx=100, Ny=40,
        notch=False,
        disps=onp.array([0.0]),
        times=onp.linspace(0.0, 0.10, 11),
        T_cold=293.15,
        max_inner=100,
        inner_tol=2.0e-5,
        description=f"P0 resolved matched quench: {_desc}; ell/h=4.",
        save_every=1,
        tags=("cpu", "T2", "P0", "resolved"),
    ))

# Mesh-density campaign at fixed material length ell=0.5 mm.  The sequence
# spans an under-resolved grid, the accepted baseline, a 16k-cell refinement,
# and the 39k-cell density used by the CMAME-scale reference deck.
for _name, _nx, _ny in (
    ("T2_mesh_050x020_k0", 50, 20),
    ("T2_mesh_200x080_k0", 200, 80),
    ("T2_mesh_312x125_k0", 312, 125),
):
    _register(CaseSpec(
        name=_name,
        mode="quench",
        mat=ceramic_from_lame(0.5, True, False),
        Lx=12.5, Ly=5.0, Nx=_nx, Ny=_ny,
        notch=False,
        disps=onp.array([0.0]),
        times=onp.linspace(0.0, 0.10, 11),
        T_cold=293.15,
        max_inner=200 if _nx >= 300 else 100,
        inner_tol=2.0e-5,
        description=(
            f"Fixed-length mesh-density study: {_nx}x{_ny} QUAD4 cells, "
            f"ell/h={0.5 / (12.5 / _nx):.2f}."
        ),
        save_every=1,
        tags=("cpu", "T2", "mesh-density", "convergence"),
    ))

# Verification and solver-comparison layer.  The affine boundary data make
# the AT2 solution spatially homogeneous, so damage and reaction admit a
# closed form and can be checked independently of crack-path selection.
for _name, _solver in (
    ("patch_staggered", "umfpack"),
    ("patch_jax", "jax-krylov"),
    ("patch_lbfgs", "lbfgs"),
    ("patch_lbfgsb", "lbfgsb"),
):
    _register(CaseSpec(
        name=_name,
        mode="patch",
        mat=_steel_at2(ell=0.04, plane_stress=True),
        Lx=1.0, Ly=1.0, Nx=32, Ny=32,
        notch=False,
        disps=onp.linspace(0.0, 6.0e-3, 13),
        times=None,
        T_cold=None,
        max_inner=80,
        inner_tol=1.0e-8,
        description=(
            "Homogeneous AT2 plane-stress patch with an analytical damage and "
            f"reaction solution; solver={_solver}."
        ),
        solver=_solver,
        save_every=3,
        tags=("cpu", "verification", "solver-benchmark"),
        lbfgs_maxiter=100,
    ))

# Backward-Euler temporal refinement at fixed geometry, material and 4,000-cell
# mesh.  The 10-step member is the accepted T2_p0_at2_k0 baseline above.
for _nstep in (5, 20, 40, 80):
    _register(CaseSpec(
        name=f"T2_time_{_nstep:03d}",
        mode="quench",
        mat=ceramic_from_lame(0.5, True, False),
        Lx=12.5, Ly=5.0, Nx=100, Ny=40,
        notch=False,
        disps=onp.array([0.0]),
        times=onp.linspace(0.0, 0.10, _nstep + 1),
        T_cold=293.15,
        max_inner=120,
        inner_tol=2.0e-5,
        description=(
            f"Fixed-mesh backward-Euler temporal refinement with {_nstep} steps "
            "to 0.1 s."
        ),
        save_every=max(1, _nstep // 10),
        tags=("cpu", "T2", "verification", "time-convergence"),
    ))

for _nstep in (20, 40):
    _register(CaseSpec(
        name=f"T2_time_{_nstep:03d}_kd",
        mode="quench",
        mat=ceramic_from_lame(0.5, True, True),
        Lx=12.5, Ly=5.0, Nx=100, Ny=40,
        notch=False,
        disps=onp.array([0.0]),
        times=onp.linspace(0.0, 0.10, _nstep + 1),
        T_cold=293.15,
        max_inner=120,
        inner_tol=2.0e-5,
        description=(
            f"{_nstep}-step matched quench with damage-dependent conductivity."
        ),
        save_every=max(1, _nstep // 10),
        tags=("cpu", "T2", "verification", "time-convergence", "matched-pair"),
    ))

# Manuscript refinement matrix. Spatial convergence is assessed at 40 time
# steps on three meshes, while the 100x40 and 200x80 meshes are repeated at
# 80 steps to expose h-dt interaction.  All damage subproblems use a
# primal-dual active set and the principal pairs hold Gc constant so that the
# conductivity feedback is not mixed with a synthetic toughness law.
for _nx, _ny, _nstep in (
    (100, 40, 40),
    (100, 40, 80),
    (200, 80, 40),
    (200, 80, 80),
    (400, 160, 40),
):
    for _suffix, _degrade in (("k0", False), ("kd", True)):
        _register(CaseSpec(
            name=f"TS_{_nx:03d}x{_ny:03d}_n{_nstep:03d}_{_suffix}",
            mode="quench",
            mat=ceramic_manuscript(0.5, _degrade),
            Lx=12.5,
            Ly=5.0,
            Nx=_nx,
            Ny=_ny,
            notch=False,
            disps=onp.array([0.0]),
            times=onp.linspace(0.0, 0.10, _nstep + 1),
            T_cold=293.15,
            max_inner=320,
            inner_tol=2.0e-5,
            description=(
                "Matched clamped ceramic quench with constant toughness, "
                "active-set irreversibility and lumped heat capacity: "
                f"{_nx}x{_ny} cells, {_nstep} steps, "
                f"{'damage-dependent' if _degrade else 'constant'} conductivity."
            ),
            save_every=max(1, _nstep // 10),
            tags=("cpu", "T2", "manuscript", "matched-convergence"),
            lump_heat_capacity=True,
            irreversibility_solver="active-set",
            solver="pyamg" if _nx == 400 else "umfpack",
        ))

# Finite-rate cooling pair at Bi=10. This pair
# changes only the conductivity law and therefore tests shielding directly.
for _suffix, _degrade in (("k0", False), ("kd", True)):
    _register(CaseSpec(
        name=f"TS_Bi10_n080_{_suffix}",
        mode="quench",
        mat=ceramic_manuscript(0.5, _degrade),
        Lx=12.5,
        Ly=5.0,
        Nx=100,
        Ny=40,
        notch=False,
        disps=onp.array([0.0]),
        times=onp.linspace(0.0, 0.10, 81),
        T_cold=293.15,
        max_inner=320,
        inner_tol=2.0e-5,
        description=f"Matched Bi=10 control with {_suffix} conductivity.",
        save_every=8,
        tags=("cpu", "T2", "manuscript", "Biot-pair"),
        thermal_bc="convective",
        h_conv=10.0 * 0.031 / 2.5,
        T_ambient=293.15,
        lump_heat_capacity=True,
        irreversibility_solver="active-set",
    ))

# Minimally constrained free-quench pair.  Two point constraints remove the
# rigid modes without clamping an entire edge.
for _suffix, _degrade in (("k0", False), ("kd", True)):
    _register(CaseSpec(
        name=f"TS_free_n080_{_suffix}",
        mode="quench",
        mat=ceramic_manuscript(0.5, _degrade),
        Lx=12.5,
        Ly=5.0,
        Nx=100,
        Ny=40,
        notch=False,
        disps=onp.array([0.0]),
        times=onp.linspace(0.0, 0.10, 81),
        T_cold=293.15,
        max_inner=180,
        inner_tol=2.0e-5,
        description=f"Minimally constrained free-quench control with {_suffix} conductivity.",
        save_every=8,
        tags=("cpu", "T2", "manuscript", "free-quench"),
        lump_heat_capacity=True,
        irreversibility_solver="active-set",
        mechanical_bc="minimal",
    ))

# Controls for the toughness law and finite AT1 initiation threshold.
for _control, _gc_T, _model in (("gcT", True, "AT2"), ("AT1", False, "AT1")):
    for _suffix, _degrade in (("k0", False), ("kd", True)):
        _register(CaseSpec(
            name=f"TS_{_control}_n080_{_suffix}",
            mode="quench",
            mat=ceramic_manuscript(0.5, _degrade, gc_T=_gc_T, model=_model),
            Lx=12.5,
            Ly=5.0,
            Nx=100,
            Ny=40,
            notch=False,
            disps=onp.array([0.0]),
            times=onp.linspace(0.0, 0.10, 81),
            T_cold=293.15,
            max_inner=320,
            inner_tol=2.0e-5,
            description=f"{_control} control with {_suffix} conductivity.",
            save_every=8,
            tags=("cpu", "T2", "manuscript", "constitutive-control"),
            lump_heat_capacity=True,
            irreversibility_solver="active-set",
        ))

# Multi-case supplementary case.  Each 2-D uniform case uses h=ell/4.
for _ell, _nx, _ny in ((1.0, 50, 20), (0.25, 200, 80)):
    for _suffix, _degrade in (("k0", False), ("kd", True)):
        _register(CaseSpec(
            name=f"T2_length_l{str(_ell).replace('.', 'p')}_{_suffix}",
            mode="quench",
            mat=ceramic_from_lame(_ell, True, _degrade),
            Lx=12.5, Ly=5.0, Nx=_nx, Ny=_ny,
            notch=False, disps=onp.array([0.0]),
            times=onp.linspace(0.0, 0.10, 11), T_cold=293.15,
            max_inner=25 if _degrade else 100, inner_tol=2.0e-5,
            description=f"Length-scale study, ell={_ell} mm, ell/h=4, {'degraded' if _degrade else 'constant'} conductivity.",
            save_every=2, tags=("T2", "expansion", "length-scale"),
        ))

for _suffix, _degrade in (("k0", False), ("kd", True)):
    _register(CaseSpec(
        name=f"T2_length_l0p75_{_suffix}", mode="quench",
        mat=ceramic_from_lame(0.75, True, _degrade),
        Lx=12.5, Ly=5.0, Nx=67, Ny=27,
        notch=False, disps=onp.array([0.0]),
        times=onp.linspace(0.0, 0.10, 11), T_cold=293.15,
        max_inner=25 if _degrade else 100, inner_tol=2.0e-5,
        description=f"Length-scale study, ell=0.75 mm, ell/h approximately 4, {'degraded' if _degrade else 'constant'} conductivity.",
        save_every=2, tags=("T2", "expansion", "length-scale"),
    ))

for _suffix, _degrade in (("k0", False), ("kd", True)):
    _register(CaseSpec(
        name=f"T2_pfczm_p0_{_suffix}", mode="quench",
        mat=ceramic_pfczm(0.5, True, _degrade, ft=0.12),
        Lx=12.5, Ly=5.0, Nx=100, Ny=40,
        notch=False, disps=onp.array([0.0]),
        times=onp.linspace(0.0, 0.10, 11), T_cold=293.15,
        max_inner=25, inner_tol=2.0e-5,
        description=f"PF-CZM strength-controlled quench, {'degraded' if _degrade else 'constant'} conductivity.",
        save_every=2, tags=("T2", "expansion", "PFCZM"),
    ))

for _suffix, _degrade in (("k0", False), ("kd", True)):
    _register(CaseSpec(
        name=f"T2_pfczm_ft0p5_{_suffix}", mode="quench",
        mat=ceramic_pfczm(0.5, True, _degrade, ft=0.5),
        Lx=12.5, Ly=5.0, Nx=100, Ny=40,
        notch=False, disps=onp.array([0.0]),
        times=onp.linspace(0.0, 0.10, 11), T_cold=293.15,
        max_inner=20, inner_tol=2.0e-5,
        description=f"PF-CZM calibrated-strength comparison, ft=0.5 GPa, {'degraded' if _degrade else 'constant'} conductivity.",
        save_every=2, tags=("T2", "expansion", "PFCZM", "strength-screen"),
    ))

for _bi in (0.1, 1.0, 10.0):
    _register(CaseSpec(
        name=f"T2_convective_Bi{str(_bi).replace('.', 'p')}", mode="quench",
        mat=ceramic_from_lame(0.5, True, False),
        Lx=12.5, Ly=5.0, Nx=100, Ny=40,
        notch=False, disps=onp.array([0.0]),
        times=onp.linspace(0.0, 0.10, 11), T_cold=293.15,
        max_inner=20, inner_tol=2.0e-5,
        description=f"Convective quench with Bi={_bi} based on Ly/2.",
        save_every=2, tags=("T2", "expansion", "convection"),
        thermal_bc="convective", h_conv=_bi * 0.031 / 2.5, T_ambient=293.15,
    ))

_register(CaseSpec(
    name="T2_notched_p0", mode="quench",
    mat=ceramic_from_lame(0.5, True, False),
    Lx=12.5, Ly=5.0, Nx=100, Ny=40,
    notch=True, notch_style="left", notch_length=2.0, notch_y=2.5,
    disps=onp.array([0.0]), times=onp.linspace(0.0, 0.10, 11), T_cold=293.15,
    max_inner=100, inner_tol=2.0e-5,
    description="Resolved quench with a 2 mm mid-height edge flaw.",
    save_every=2, tags=("T2", "expansion", "defect"),
))

for _kr in (1.0e-2, 1.0e-3):
    _register(CaseSpec(
        name=f"T2_kres_{_kr:.0e}".replace("-", "m"), mode="quench",
        mat=replace(ceramic_from_lame(0.5, True, True), thermal_residual_k=_kr),
        Lx=12.5, Ly=5.0, Nx=100, Ny=40,
        notch=False, disps=onp.array([0.0]),
        times=onp.linspace(0.0, 0.10, 11), T_cold=293.15,
        max_inner=20, inner_tol=2.0e-5,
        description=f"Residual-conductivity factor design, kappa residual ratio={_kr:g}.",
        save_every=2, tags=("T2", "expansion", "factor-design"),
    ))

_register(CaseSpec(
    name="T2_reheat_p0", mode="quench",
    mat=ceramic_from_lame(0.5, True, False),
    Lx=12.5, Ly=5.0, Nx=100, Ny=40,
    notch=False, disps=onp.array([0.0]),
    times=onp.linspace(0.0, 0.10, 11), T_cold=293.15,
    max_inner=100, inner_tol=2.0e-5,
    description="Resolved quench-reheat irreversibility control.",
    save_every=1, tags=("T2", "expansion", "cycle"), T_reheat=0.06,
))

_register(CaseSpec(
    name="T2_amr_p0", mode="quench",
    mat=ceramic_from_lame(0.5, True, False),
    Lx=12.5, Ly=5.0, Nx=25, Ny=10,
    notch=False, disps=onp.array([0.0]),
    times=onp.linspace(0.0, 0.10, 11), T_cold=293.15,
    max_inner=20, inner_tol=2.0e-5,
    description="Conforming TRI3 AMR quench with local ell/h target 4.",
    save_every=2, tags=("T2", "expansion", "amr"),
    amr=True, amr_max_level=5, amr_every=1, amr_d_th=0.10,
    amr_h_over_ell=0.25, amr_buffer=1, amr_max_cells=12000, ele_type="TRI3",
))

_register(CaseSpec(
    name="T3_resolved_path", mode="quench3d",
    mat=ceramic_from_lame(2.5, True, False, plane_stress=False),
    Lx=12.5, Ly=5.0, Lz=2.5, Nx=20, Ny=8, Nz=4,
    notch=False, disps=onp.array([0.0]),
    times=onp.linspace(0.0, 0.05, 6), T_cold=293.15,
    max_inner=20, inner_tol=3.0e-5,
    description="Reduced HEX8 three-dimensional consistency path with ell/h=4.",
    save_every=1, tags=("T3", "expansion", "3d"), ele_type="HEX8",
))

_register(CaseSpec(
    name="T2_quench_ref",
    mode="quench",
    mat=ceramic_from_lame(ell=0.2, gc_T=True, degrade_k=True),
    Lx=25.0, Ly=10.0, Nx=125, Ny=50,
    notch=False,
    disps=onp.array([0.0]),
    times=onp.linspace(0.0, 0.20, 201),
    T_cold=293.15,
    max_inner=4,
    inner_tol=1e-6,
    description="Production half-plate quench matching Jin case 6 (GPU/MPI).",
    save_every=10,
    tags=("gpu", "T2", "production"),
))

_register(CaseSpec(
    name="M1_tension_amr",
    mode="tension",
    mat=_steel_at2(ell=0.015, plane_stress=True),
    Lx=1.0, Ly=1.0, Nx=32, Ny=32,
    notch=True,
    disps=_linspace_load(0.0, 4.0e-3, 7.5e-3, 9, 29),
    times=None,
    T_cold=None,
    max_inner=3,
    inner_tol=1e-5,
    description="AMR tension: coarse 32x32 TRI3, longest-edge bisection to ℓ/h≥4 in the crack band.",
    save_every=2,
    tags=("cpu", "M1", "amr"),
    amr=True,
    amr_max_level=5,
    amr_every=2,
    amr_d_th=0.12,
    amr_h_over_ell=0.25,
    amr_buffer=1,
    amr_max_cells=14000,
    ele_type="TRI3",
))

_register(CaseSpec(
    name="M1_shear_amr",
    mode="shear",
    mat=_steel_at2(ell=0.015, plane_stress=True),
    Lx=1.0, Ly=1.0, Nx=32, Ny=32,
    notch=True,
    disps=_linspace_load(0.0, 8.0e-3, 1.5e-2, 9, 25),
    times=None,
    T_cold=None,
    max_inner=3,
    inner_tol=1e-5,
    description="AMR shear: damage-driven longest-edge bisection, curved path.",
    save_every=2,
    tags=("cpu", "M1", "amr"),
    amr=True,
    amr_max_level=5,
    amr_every=2,
    amr_d_th=0.12,
    amr_h_over_ell=0.25,
    amr_buffer=1,
    amr_max_cells=14000,
    ele_type="TRI3",
))

_register(CaseSpec(
    name="T2_quench_amr",
    mode="quench",
    mat=ceramic_from_lame(ell=0.2, gc_T=True, degrade_k=True),
    Lx=12.5, Ly=5.0, Nx=40, Ny=16,
    notch=False,
    disps=onp.array([0.0]),
    times=onp.linspace(0.0, 0.12, 25),
    T_cold=293.15,
    max_inner=2,
    inner_tol=1e-5,
    description="AMR quench: thermal-gradient + damage indicators, κ(d) and Gc(T).",
    save_every=2,
    tags=("cpu", "T2", "amr"),
    amr=True,
    amr_max_level=4,
    amr_every=2,
    amr_d_th=0.10,
    amr_h_over_ell=0.25,
    amr_buffer=1,
    amr_max_cells=16000,
    ele_type="TRI3",
))

_register(CaseSpec(
    name="M2_cyclic_cpu",
    mode="shear",
    mat=_steel_at2(ell=0.04, plane_stress=True),
    Lx=1.0, Ly=1.0, Nx=40, Ny=40,
    notch=True,
    disps=onp.hstack((
        onp.linspace(0.0, 1.0e-2, 16, endpoint=False),
        onp.linspace(1.0e-2, 2.0e-3, 9, endpoint=False),
        onp.linspace(2.0e-3, 1.2e-2, 16),
    )),
    times=None,
    T_cold=None,
    max_inner=2,
    inner_tol=1e-5,
    description="Cyclic shear: load–unload–reload for history-field irreversibility.",
    save_every=2,
    tags=("cpu", "M2"),
))

_register(CaseSpec(
    name="M1_shear_ref",
    mode="shear",
    mat=_steel_at2(ell=0.015, plane_stress=True),
    Lx=1.0, Ly=1.0, Nx=80, Ny=80,
    notch=True,
    disps=_linspace_load(0.0, 8.0e-3, 1.5e-2, 9, 31),
    times=None,
    T_cold=None,
    max_inner=3,
    inner_tol=1e-6,
    description="Production-scale uniform shear, same ℓ as tension_ref, for AMR comparison.",
    save_every=2,
    tags=("cpu", "M1"),
))

_register(CaseSpec(
    name="M1_tension_at1_cpu",
    mode="tension",
    mat=_steel_at1(ell=0.04, plane_stress=True),
    Lx=1.0, Ly=1.0, Nx=40, Ny=40,
    notch=True,
    disps=_linspace_load(0.0, 4.0e-3, 6.5e-3, 11, 21),
    times=None,
    T_cold=None,
    max_inner=2,
    inner_tol=1e-5,
    description="CPU AT1 tension on the 40x40 Miehe square.",
    save_every=2,
    tags=("cpu", "M1", "AT1"),
))

_register(CaseSpec(
    name="T1_thermo_shear_cpu",
    mode="shear",
    mat=_steel_thermo(ell=0.04),
    Lx=1.0, Ly=1.0, Nx=40, Ny=40,
    notch=True,
    disps=onp.linspace(0.0, 1.2e-2, 25),
    times=None,
    T_cold=None,
    max_inner=2,
    inner_tol=1e-5,
    description="T1: top-edge shear plus a vertical temperature gradient with Gc(T).",
    save_every=2,
    tags=("cpu", "T1"),
    T_top=573.15,
    T_bot=293.15,
))

_register(CaseSpec(
    name="M1_tension_lbfgs",
    mode="tension",
    mat=_steel_at2(ell=0.04, plane_stress=True),
    Lx=1.0, Ly=1.0, Nx=40, Ny=40,
    notch=True,
    disps=onp.linspace(0.0, 6.5e-3, 17),
    times=None,
    T_cold=None,
    max_inner=1,
    inner_tol=1e-5,
    description="CPU L-BFGS notched tension on the 40x40 Miehe square.",
    solver="lbfgs",
    save_every=4,
    tags=("cpu", "M1", "lbfgs"),
    lbfgs_maxiter=30,
))

_register(CaseSpec(
    name="M1_shear_lbfgs",
    mode="shear",
    mat=_steel_at2(ell=0.04, plane_stress=True),
    Lx=1.0, Ly=1.0, Nx=40, Ny=40,
    notch=True,
    disps=onp.linspace(0.0, 1.2e-2, 17),
    times=None,
    T_cold=None,
    max_inner=1,
    inner_tol=1e-5,
    description="CPU L-BFGS notched shear on the 40x40 Miehe square.",
    solver="lbfgs",
    save_every=4,
    tags=("cpu", "M1", "lbfgs"),
    lbfgs_maxiter=30,
))

_register(CaseSpec(
    name="M2_cyclic_lbfgsb",
    mode="shear",
    mat=_steel_at2(ell=0.04, plane_stress=True),
    Lx=1.0, Ly=1.0, Nx=40, Ny=40,
    notch=True,
    disps=onp.hstack((
        onp.linspace(0.0, 1.0e-2, 9, endpoint=False),
        onp.linspace(1.0e-2, 2.0e-3, 5, endpoint=False),
        onp.linspace(2.0e-3, 1.2e-2, 9),
    )),
    times=None,
    T_cold=None,
    max_inner=1,
    inner_tol=1e-5,
    description="CPU L-BFGS-B cyclic shear with box-constraint irreversibility.",
    solver="lbfgsb",
    save_every=4,
    tags=("cpu", "M2", "lbfgs"),
    lbfgs_maxiter=25,
))

_register(CaseSpec(
    name="T2_quench_lbfgs_k0",
    mode="quench",
    mat=ceramic_from_lame(ell=0.4, gc_T=False, degrade_k=False),
    Lx=12.5, Ly=5.0, Nx=40, Ny=16,
    notch=False,
    disps=onp.array([0.0]),
    times=onp.linspace(0.0, 0.06, 13),
    T_cold=293.15,
    max_inner=1,
    inner_tol=1e-5,
    description="CPU L-BFGS quench, kappa0 and constant Gcz.",
    solver="lbfgs",
    save_every=3,
    tags=("cpu", "T2", "lbfgs"),
    lbfgs_maxiter=25,
))

_register(CaseSpec(
    name="T2_quench_lbfgs_kd",
    mode="quench",
    mat=ceramic_from_lame(ell=0.4, gc_T=True, degrade_k=True),
    Lx=12.5, Ly=5.0, Nx=40, Ny=16,
    notch=False,
    disps=onp.array([0.0]),
    times=onp.linspace(0.0, 0.06, 13),
    T_cold=293.15,
    max_inner=1,
    inner_tol=1e-5,
    description="CPU L-BFGS quench, kappa(d) and Gc(T).",
    solver="lbfgs",
    save_every=3,
    tags=("cpu", "T2", "lbfgs"),
    lbfgs_maxiter=25,
))

_register(CaseSpec(
    name="T2_quench_pfczm_lbfgs",
    mode="quench",
    mat=ceramic_pfczm(ell=0.4, gc_T=True, degrade_k=True, ft=0.12),
    Lx=12.5, Ly=5.0, Nx=40, Ny=16,
    notch=False,
    disps=onp.array([0.0]),
    times=onp.linspace(0.0, 0.06, 13),
    T_cold=293.15,
    max_inner=1,
    inner_tol=1e-5,
    description="CPU L-BFGS PF-CZM quench with kappa(d) and Gc(T).",
    solver="lbfgs",
    save_every=3,
    tags=("cpu", "T2", "lbfgs", "PFCZM"),
    lbfgs_maxiter=25,
))

_register(CaseSpec(
    name="T2_quench_reheat_lbfgsb",
    mode="quench",
    mat=ceramic_from_lame(ell=0.4, gc_T=True, degrade_k=True),
    Lx=12.5, Ly=5.0, Nx=40, Ny=16,
    notch=False,
    disps=onp.array([0.0]),
    times=onp.unique(onp.hstack((
        onp.linspace(0.0, 0.05, 11),
        onp.linspace(0.055, 0.08, 6),
    ))),
    T_cold=293.15,
    max_inner=1,
    inner_tol=1e-5,
    description="CPU L-BFGS-B quench then reheat; box constraints, no history field.",
    solver="lbfgsb",
    save_every=3,
    tags=("cpu", "T2", "lbfgs"),
    lbfgs_maxiter=25,
    T_reheat=0.052,
))

_register(CaseSpec(
    name="T3_shock_lbfgs",
    mode="quench3d",
    mat=ceramic_from_lame(ell=0.5, gc_T=False, degrade_k=False, plane_stress=False),
    Lx=12.5, Ly=5.0, Nx=8, Ny=4,
    notch=False,
    disps=onp.array([0.0]),
    times=onp.linspace(0.0, 0.04, 7),
    T_cold=293.15,
    max_inner=1,
    inner_tol=1e-5,
    description="CPU L-BFGS HEX8 thermal shock on a reduced 3D brick (not the 1e7-DOF production mesh).",
    solver="lbfgs",
    save_every=2,
    tags=("cpu", "T3", "lbfgs"),
    ele_type="HEX8",
    Nz=3,
    Lz=2.5,
    lbfgs_maxiter=20,
))

_register(CaseSpec(
    name="I1_gcT_inverse",
    mode="inverse",
    mat=ceramic_from_lame(ell=0.8, gc_T=True, degrade_k=False),
    Lx=12.5, Ly=5.0, Nx=25, Ny=10,
    notch=False,
    disps=onp.array([0.0]),
    times=onp.linspace(0.0, 0.04, 9),
    T_cold=293.15,
    max_inner=1,
    inner_tol=1e-4,
    description="I1: FD outer L-BFGS identification of (Gc0, b1, b2) from a near-edge damage snapshot.",
    solver="umfpack",
    save_every=50,
    tags=("cpu", "I1"),
    lbfgs_maxiter=4,
    theta_true=(42.47e-6, 1.8, 1.1),
))

# --- Reference-case CPU layer: same residuals, resolved enough for the claims -----
# M1 production L-BFGS on the 80x80 mesh already used for staggered AT2.
_register(CaseSpec(
    name="M1_tension_lbfgs_ref",
    mode="tension",
    mat=_steel_at2(ell=0.015, plane_stress=True),
    Lx=1.0, Ly=1.0, Nx=80, Ny=80,
    notch=True,
    disps=_linspace_load(0.0, 4.0e-3, 8.0e-3, 9, 25),
    times=None,
    T_cold=None,
    max_inner=1,
    inner_tol=1e-6,
    description="L-BFGS tension on the 80x80 production mesh, u to 0.008 mm.",
    solver="lbfgs",
    save_every=4,
    tags=("cpu", "M1", "lbfgs", "reference"),
    lbfgs_maxiter=28,
))

_register(CaseSpec(
    name="M1_shear_lbfgs_ref",
    mode="shear",
    mat=_steel_at2(ell=0.015, plane_stress=True),
    Lx=1.0, Ly=1.0, Nx=80, Ny=80,
    notch=True,
    disps=_linspace_load(0.0, 8.0e-3, 2.0e-2, 9, 25),
    times=None,
    T_cold=None,
    max_inner=1,
    inner_tol=1e-6,
    description="L-BFGS shear on the 80x80 production mesh, u to 0.020 mm.",
    solver="lbfgs",
    save_every=4,
    tags=("cpu", "M1", "lbfgs", "reference"),
    lbfgs_maxiter=28,
))

_register(CaseSpec(
    name="M1_shear_ref_ext",
    mode="shear",
    mat=_steel_at2(ell=0.015, plane_stress=True),
    Lx=1.0, Ly=1.0, Nx=80, Ny=80,
    notch=True,
    disps=_linspace_load(0.0, 8.0e-3, 2.0e-2, 9, 33),
    times=None,
    T_cold=None,
    max_inner=3,
    inner_tol=1e-6,
    description="Staggered 80x80 shear continued to u=0.020 mm for a developed process zone.",
    save_every=2,
    tags=("cpu", "M1", "reference"),
))

_register(CaseSpec(
    name="M1_tension_at1_ref",
    mode="tension",
    mat=_steel_at1(ell=0.015, plane_stress=True),
    Lx=1.0, Ly=1.0, Nx=80, Ny=80,
    notch=True,
    disps=_linspace_load(0.0, 4.0e-3, 7.0e-3, 9, 25),
    times=None,
    T_cold=None,
    max_inner=3,
    inner_tol=1e-6,
    description="AT1 tension on the 80x80 production mesh.",
    save_every=4,
    tags=("cpu", "M1", "AT1", "reference"),
))

_register(CaseSpec(
    name="T1_thermo_shear_ref",
    mode="shear",
    mat=_steel_thermo(ell=0.015),
    Lx=1.0, Ly=1.0, Nx=80, Ny=80,
    notch=True,
    disps=_linspace_load(0.0, 8.0e-3, 1.5e-2, 9, 25),
    times=None,
    T_cold=None,
    max_inner=3,
    inner_tol=1e-6,
    description="T1 production: 80x80 shear plus vertical temperature gradient.",
    save_every=4,
    tags=("cpu", "T1", "reference"),
    T_top=573.15,
    T_bot=293.15,
))

# Pure conductivity L-BFGS pair: same deck as staggered T2_quench_cpu / _kappa.
_register(CaseSpec(
    name="T2_quench_lbfgs_k0gct",
    mode="quench",
    mat=ceramic_from_lame(ell=0.4, gc_T=True, degrade_k=False),
    Lx=12.5, Ly=5.0, Nx=40, Ny=16,
    notch=False,
    disps=onp.array([0.0]),
    times=onp.linspace(0.0, 0.10, 21),
    T_cold=293.15,
    max_inner=1,
    inner_tol=1e-5,
    description="L-BFGS quench, kappa0 and Gc(T), matching staggered T2_quench_cpu.",
    solver="lbfgs",
    save_every=2,
    tags=("cpu", "T2", "lbfgs", "reference"),
    lbfgs_maxiter=25,
))

_register(CaseSpec(
    name="T2_quench_lbfgs_kd10",
    mode="quench",
    mat=ceramic_from_lame(ell=0.4, gc_T=True, degrade_k=True),
    Lx=12.5, Ly=5.0, Nx=40, Ny=16,
    notch=False,
    disps=onp.array([0.0]),
    times=onp.linspace(0.0, 0.10, 21),
    T_cold=293.15,
    max_inner=1,
    inner_tol=1e-5,
    description="L-BFGS quench, kappa(d) and Gc(T), matching staggered T2_quench_cpu_kappa.",
    solver="lbfgs",
    save_every=2,
    tags=("cpu", "T2", "lbfgs", "reference"),
    lbfgs_maxiter=25,
))

_register(CaseSpec(
    name="T2_quench_pfczm_ref",
    mode="quench",
    mat=ceramic_pfczm(ell=0.2, gc_T=True, degrade_k=True, ft=0.12),
    Lx=12.5, Ly=5.0, Nx=80, Ny=32,
    notch=False,
    disps=onp.array([0.0]),
    times=onp.linspace(0.0, 0.10, 21),
    T_cold=293.15,
    max_inner=1,
    inner_tol=1e-5,
    description="PF-CZM quench on 80x32 (ell=0.2 mm) with kappa(d) and Gc(T).",
    solver="lbfgs",
    save_every=4,
    tags=("cpu", "T2", "lbfgs", "PFCZM", "reference"),
    lbfgs_maxiter=25,
))

_register(CaseSpec(
    name="T3_shock_lbfgs_mid",
    mode="quench3d",
    mat=ceramic_from_lame(ell=0.35, gc_T=False, degrade_k=True, plane_stress=False),
    Lx=12.5, Ly=5.0, Nx=16, Ny=8,
    notch=False,
    disps=onp.array([0.0]),
    times=onp.linspace(0.0, 0.08, 13),
    T_cold=293.15,
    max_inner=1,
    inner_tol=1e-5,
    description="Reduced 16×8×5 HEX8 thermal-shock calculation with damage-dependent conductivity.",
    solver="lbfgs",
    save_every=2,
    tags=("cpu", "T3", "lbfgs", "reference"),
    ele_type="HEX8",
    Nz=5,
    Lz=2.5,
    lbfgs_maxiter=22,
))

_register(CaseSpec(
    name="I1_gcT_inverse_cpu",
    mode="inverse",
    mat=ceramic_from_lame(ell=0.4, gc_T=True, degrade_k=False),
    Lx=12.5, Ly=5.0, Nx=40, Ny=16,
    notch=False,
    disps=onp.array([0.0]),
    times=onp.linspace(0.0, 0.08, 17),
    T_cold=293.15,
    max_inner=2,
    inner_tol=1e-5,
    description="I1 on the 40x16 T2-CPU mesh: FD outer L-BFGS for (Gc0, b1, b2).",
    solver="umfpack",
    save_every=50,
    tags=("cpu", "I1", "reference"),
    lbfgs_maxiter=4,
    theta_true=(42.47e-6, 1.8, 1.1),
))

# --- CMAME process-zone density: ℓ/h ≳ 4 and O(10^4) cells -----------------
# Miehe 2010 CMAME resolved the 1 mm square at ℓ=0.015 mm with ~2×10^4 TRI3.
# Wu/Jin CMAME decks keep four to five elements across ℓ. Uniform 80×80
# (ℓ/h=1.2) and T2 40×16 (ℓ/h=1.28) do not meet that rule.
_register(CaseSpec(
    name="M1_tension_cmame",
    mode="tension",
    mat=_steel_at2(ell=0.015, plane_stress=True),
    Lx=1.0, Ly=1.0, Nx=256, Ny=256,
    notch=True,
    disps=_linspace_load(0.0, 4.0e-3, 8.0e-3, 9, 25),
    times=None,
    T_cold=None,
    max_inner=3,
    inner_tol=1e-6,
    description="CMAME-density AT2 tension: 256×256 QUAD4, ℓ/h=3.84, 65536 cells.",
    save_every=6,
    tags=("cpu", "M1", "cmame"),
))

_register(CaseSpec(
    name="M1_shear_cmame",
    mode="shear",
    mat=_steel_at2(ell=0.015, plane_stress=True),
    Lx=1.0, Ly=1.0, Nx=256, Ny=256,
    notch=True,
    disps=_linspace_load(0.0, 8.0e-3, 2.0e-2, 9, 25),
    times=None,
    T_cold=None,
    max_inner=3,
    inner_tol=1e-6,
    description="CMAME-density AT2 shear: 256×256 QUAD4 to u=0.020 mm.",
    save_every=6,
    tags=("cpu", "M1", "cmame"),
))

_register(CaseSpec(
    name="M1_tension_lbfgs_cmame",
    mode="tension",
    mat=_steel_at2(ell=0.015, plane_stress=True),
    Lx=1.0, Ly=1.0, Nx=256, Ny=256,
    notch=True,
    disps=_linspace_load(0.0, 4.0e-3, 8.0e-3, 9, 25),
    times=None,
    T_cold=None,
    max_inner=1,
    inner_tol=1e-6,
    description="L-BFGS on the 256×256 CMAME tension residual.",
    solver="lbfgs",
    save_every=6,
    tags=("cpu", "M1", "lbfgs", "cmame"),
    lbfgs_maxiter=24,
))

_register(CaseSpec(
    name="M1_shear_lbfgs_cmame",
    mode="shear",
    mat=_steel_at2(ell=0.015, plane_stress=True),
    Lx=1.0, Ly=1.0, Nx=256, Ny=256,
    notch=True,
    disps=_linspace_load(0.0, 8.0e-3, 2.0e-2, 9, 25),
    times=None,
    T_cold=None,
    max_inner=1,
    inner_tol=1e-6,
    description="L-BFGS on the 256×256 CMAME shear residual.",
    solver="lbfgs",
    save_every=6,
    tags=("cpu", "M1", "lbfgs", "cmame"),
    lbfgs_maxiter=24,
))

_register(CaseSpec(
    name="T2_quench_cmame_k0",
    mode="quench",
    mat=ceramic_from_lame(ell=0.2, gc_T=True, degrade_k=False),
    Lx=12.5, Ly=5.0, Nx=312, Ny=125,
    notch=False,
    disps=onp.array([0.0]),
    times=onp.linspace(0.0, 0.10, 21),
    T_cold=293.15,
    max_inner=3,
    inner_tol=1e-6,
    description="CMAME-density quench κ0, Gc(T): 312×125, ℓ/h=5, 39000 cells.",
    save_every=4,
    tags=("cpu", "T2", "cmame"),
))

_register(CaseSpec(
    name="T2_quench_cmame_kd",
    mode="quench",
    mat=ceramic_from_lame(ell=0.2, gc_T=True, degrade_k=True),
    Lx=12.5, Ly=5.0, Nx=312, Ny=125,
    notch=False,
    disps=onp.array([0.0]),
    times=onp.linspace(0.0, 0.10, 21),
    T_cold=293.15,
    max_inner=3,
    inner_tol=1e-6,
    description="CMAME-density quench κ(d), Gc(T): same 312×125 residual as k0.",
    save_every=4,
    tags=("cpu", "T2", "cmame"),
))

_register(CaseSpec(
    name="T2_quench_lbfgs_cmame_k0",
    mode="quench",
    mat=ceramic_from_lame(ell=0.2, gc_T=True, degrade_k=False),
    Lx=12.5, Ly=5.0, Nx=312, Ny=125,
    notch=False,
    disps=onp.array([0.0]),
    times=onp.linspace(0.0, 0.10, 21),
    T_cold=293.15,
    max_inner=1,
    inner_tol=1e-6,
    description="L-BFGS CMAME quench, κ0 and Gc(T), matching T2_quench_cmame_k0.",
    solver="lbfgs",
    save_every=4,
    tags=("cpu", "T2", "lbfgs", "cmame"),
    lbfgs_maxiter=22,
))

_register(CaseSpec(
    name="T2_quench_lbfgs_cmame_kd",
    mode="quench",
    mat=ceramic_from_lame(ell=0.2, gc_T=True, degrade_k=True),
    Lx=12.5, Ly=5.0, Nx=312, Ny=125,
    notch=False,
    disps=onp.array([0.0]),
    times=onp.linspace(0.0, 0.10, 21),
    T_cold=293.15,
    max_inner=1,
    inner_tol=1e-6,
    description="L-BFGS CMAME quench, κ(d) and Gc(T), matching T2_quench_cmame_kd.",
    solver="lbfgs",
    save_every=4,
    tags=("cpu", "T2", "lbfgs", "cmame"),
    lbfgs_maxiter=22,
))

_register(CaseSpec(
    name="T2_quench_pfczm_cmame",
    mode="quench",
    mat=ceramic_pfczm(ell=0.2, gc_T=True, degrade_k=True, ft=0.12),
    Lx=12.5, Ly=5.0, Nx=312, Ny=125,
    notch=False,
    disps=onp.array([0.0]),
    times=onp.linspace(0.0, 0.10, 21),
    T_cold=293.15,
    max_inner=1,
    inner_tol=1e-6,
    description="PF-CZM quench on the 312×125 CMAME mesh (ℓ/h=5).",
    solver="lbfgs",
    save_every=4,
    tags=("cpu", "T2", "lbfgs", "PFCZM", "cmame"),
    lbfgs_maxiter=22,
))

_register(CaseSpec(
    name="M1_tension_amr_cmame",
    mode="tension",
    mat=_steel_at2(ell=0.015, plane_stress=True),
    Lx=1.0, Ly=1.0, Nx=80, Ny=80,
    notch=True,
    disps=_linspace_load(0.0, 4.0e-3, 8.0e-3, 9, 25),
    times=None,
    T_cold=None,
    max_inner=3,
    inner_tol=1e-6,
    description="AMR tension to ~50k TRI3, target h/ℓ=0.20 (ℓ/h=5) in the band.",
    save_every=6,
    tags=("cpu", "M1", "amr", "cmame"),
    amr=True,
    amr_max_level=6,
    amr_every=2,
    amr_d_th=0.10,
    amr_h_over_ell=0.20,
    amr_buffer=1,
    amr_max_cells=50000,
    ele_type="TRI3",
))

_register(CaseSpec(
    name="T2_quench_amr_cmame",
    mode="quench",
    mat=ceramic_from_lame(ell=0.2, gc_T=True, degrade_k=True),
    Lx=12.5, Ly=5.0, Nx=100, Ny=40,
    notch=False,
    disps=onp.array([0.0]),
    times=onp.linspace(0.0, 0.10, 21),
    T_cold=293.15,
    max_inner=3,
    inner_tol=1e-6,
    description="AMR quench to ~50k TRI3, thermal-skin target h/ℓ=0.20.",
    save_every=4,
    tags=("cpu", "T2", "amr", "cmame"),
    amr=True,
    amr_max_level=6,
    amr_every=2,
    amr_d_th=0.10,
    amr_h_over_ell=0.20,
    amr_buffer=1,
    amr_max_cells=50000,
    ele_type="TRI3",
))

_register(CaseSpec(
    name="T3_shock_cmame",
    mode="quench3d",
    mat=ceramic_from_lame(ell=0.25, gc_T=False, degrade_k=True, plane_stress=False),
    Lx=12.5, Ly=5.0, Nx=40, Ny=16,
    notch=False,
    disps=onp.array([0.0]),
    times=onp.linspace(0.0, 0.08, 13),
    T_cold=293.15,
    max_inner=1,
    inner_tol=1e-5,
    description="Reduced HEX8 40×16×10 thermal-shock calculation for comparison with the large-scale reference.",
    solver="lbfgs",
    save_every=2,
    tags=("cpu", "T3", "lbfgs", "cmame"),
    ele_type="HEX8",
    Nz=10,
    Lz=2.5,
    lbfgs_maxiter=20,
))

_E_lshape, _nu_lshape = _lame_to_Enu(6.16, 10.95)

_register(CaseSpec(
    name="M3_tpb",
    mode="tpb",
    mat=Material(E=20.8, nu=0.3, Gc0=5.0e-4, ell=0.0075, plane_stress=False, model="AT2"),
    Lx=8.0, Ly=2.0, Nx=64, Ny=16,
    notch=True,
    notch_style="tpb",
    disps=onp.linspace(0.0, -0.04, 21),
    times=None,
    T_cold=None,
    max_inner=3,
    inner_tol=1e-6,
    description="Three-point bending, Jin 2024: 8x2 mm, triangular notch to y=0.4, AT2, plane strain, AMR band.",
    save_every=4,
    tags=("cpu", "M3", "benchmark"),
    amr=True,
    amr_max_level=6,
    amr_every=2,
    amr_d_th=0.10,
    amr_h_over_ell=0.25,
    amr_max_cells=45000,
    load_half=0.125,
))

_register(CaseSpec(
    name="M4_lshape",
    mode="lshape",
    mat=Material(
        E=_E_lshape, nu=_nu_lshape, Gc0=9.0e-5, ell=5.0, plane_stress=True, model="PFCZM",
        ft=2.7e-3, p=2.5, a2=0.1748, a3=0.0,
    ),
    Lx=500.0, Ly=500.0, Nx=200, Ny=200,
    notch=False,
    notch_style="none",
    geometry="lshape",
    disps=onp.linspace(0.0, 0.50, 21),
    times=None,
    T_cold=None,
    max_inner=3,
    inner_tol=1e-6,
    description="Winkler L-panel, Jin 2024 PF-CZM: bottom fixed, uy at (470, 250).",
    save_every=4,
    tags=("cpu", "M4", "benchmark"),
    load_half=2.5,
    load_x=470.0,
    load_y=250.0,
))

_register(CaseSpec(
    name="M5_hole",
    mode="hole",
    mat=_steel_at2(ell=0.04),
    Lx=2.0, Ly=2.0, Nx=160, Ny=160,
    notch=True,
    notch_style="left",
    notch_length=0.60,
    notch_y=1.0,
    geometry="hole",
    hole_cx=1.30,
    hole_cy=1.0,
    hole_r=0.20,
    disps=onp.linspace(0.0, 0.012, 21),
    times=None,
    T_cold=None,
    max_inner=3,
    inner_tol=1e-6,
    description="Ambati-type notched plate with an eccentric hole, AT2, ell/h=3.2.",
    save_every=4,
    tags=("cpu", "M5", "benchmark"),
))

_register(CaseSpec(
    name="M1_shear_pfczm",
    mode="shear",
    mat=_steel_pfczm(ell=0.015),
    Lx=1.0, Ly=1.0, Nx=80, Ny=80,
    notch=True,
    disps=_linspace_load(0.0, 0.010, 0.020, 8, 12),
    times=None,
    T_cold=None,
    max_inner=3,
    inner_tol=1e-6,
    description="Isothermal PF-CZM shear on the Miehe square, monolithic L-BFGS, 80x80.",
    solver="lbfgs",
    lbfgs_maxiter=40,
    save_every=4,
    tags=("cpu", "M1", "pfczm", "benchmark"),
))


def list_cases(tag: str | None = None) -> list[CaseSpec]:
    if tag is None:
        return list(CASES.values())
    return [c for c in CASES.values() if tag in c.tags]
