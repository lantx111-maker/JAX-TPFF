# Case organization

Each public case appears exactly once in a primary category.

## Verification and solver checks

- `heat_verification` - manuscript-result: Transient heat-conduction verification against the analytical reference.
- `patch_jax` - result-available: Homogeneous AT2 plane-stress patch with an analytical damage and reaction solution; solver=jax-krylov.
- `patch_lbfgs` - result-available: Homogeneous AT2 plane-stress patch with an analytical damage and reaction solution; solver=lbfgs.
- `patch_lbfgsb` - result-available: Homogeneous AT2 plane-stress patch with an analytical damage and reaction solution; solver=lbfgsb.
- `patch_staggered` - result-available: Homogeneous AT2 plane-stress patch with an analytical damage and reaction solution; solver=umfpack.
- `plane_stress_verification` - manuscript-result: Plane-stress constitutive and finite-element verification.
- `profile_transport` - manuscript-result: Stationary AT2 profile and diffuse-layer heat-flux convergence.

## Manuscript thermal-shock cases

- `TS_100x040_n040_k0` - manuscript-result: Matched thermal-shock calculation used in the manuscript.
- `TS_100x040_n040_kd` - manuscript-result: Matched thermal-shock calculation used in the manuscript.
- `TS_100x040_n080_k0` - manuscript-result: Matched thermal-shock calculation used in the manuscript.
- `TS_100x040_n080_kd` - manuscript-result: Matched thermal-shock calculation used in the manuscript.
- `TS_200x080_n040_k0` - manuscript-result: Matched thermal-shock calculation used in the manuscript.
- `TS_200x080_n040_kd` - manuscript-result: Matched thermal-shock calculation used in the manuscript.
- `TS_200x080_n080_k0` - manuscript-result: Matched thermal-shock calculation used in the manuscript.
- `TS_200x080_n080_kd` - manuscript-result: Matched thermal-shock calculation used in the manuscript.
- `TS_400x160_n040_k0` - manuscript-result: Matched thermal-shock calculation used in the manuscript.
- `TS_400x160_n040_kd` - manuscript-result: Matched thermal-shock calculation used in the manuscript.
- `TS_AT1_n080_k0` - manuscript-result: Matched thermal-shock calculation used in the manuscript.
- `TS_AT1_n080_kd` - manuscript-result: Matched thermal-shock calculation used in the manuscript.
- `TS_Bi10_n080_k0` - manuscript-result: Matched thermal-shock calculation used in the manuscript.
- `TS_Bi10_n080_kd` - manuscript-result: Matched thermal-shock calculation used in the manuscript.
- `TS_free_n080_k0` - manuscript-result: Matched thermal-shock calculation used in the manuscript.
- `TS_free_n080_kd` - manuscript-result: Matched thermal-shock calculation used in the manuscript.
- `TS_gcT_n080_k0` - manuscript-result: Matched thermal-shock calculation used in the manuscript.
- `TS_gcT_n080_kd` - manuscript-result: Matched thermal-shock calculation used in the manuscript.

## Smoke and low-cost development cases

- `M1_shear_smoke` - result-available: CPU smoke: notched shear, AT2.
- `M1_tension_smoke` - result-available: CPU smoke: notched tension, AT2, h/ℓ≈0.63, 12 load steps.
- `T2_quench_smoke` - result-available: CPU smoke quench: half-scale plate, ℓ=0.8 mm, 10 thermal steps.

## Mechanical fracture benchmarks

- `M1_shear_cpu` - result-available: CPU standard shear. Jin simple-shear deck, coarsened ℓ.
- `M1_shear_ref` - result-available: Production-scale uniform shear, same ℓ as tension_ref, for AMR comparison.
- `M1_shear_ref_ext` - result-available: Staggered 80x80 shear continued to u=0.020 mm for a developed process zone.
- `M1_tension_cpu` - result-available: CPU standard tension. Jin/Miehe unit square, coarsened ℓ=0.04 mm.
- `M1_tension_ref` - result-available: Production tension: ℓ=0.015, h=ℓ/1.2. GPU recommended.
- `M2_cyclic_cpu` - result-available: Cyclic shear: load–unload–reload for history-field irreversibility.
- `M3_tpb` - result-available: Three-point bending, Jin 2024: 8x2 mm, triangular notch to y=0.4, AT2, plane strain, AMR band.
- `M5_hole` - result-available: Ambati-type notched plate with an eccentric hole, AT2, ell/h=3.2.

## Thermomechanical shear

- `T1_thermo_shear_cpu` - result-available: T1: top-edge shear plus a vertical temperature gradient with Gc(T).
- `T1_thermo_shear_ref` - result-available: T1 production: 80x80 shear plus vertical temperature gradient.

## Two-dimensional thermal shock

- `T2_quench_cpu` - result-available: CPU quench with temperature-dependent toughness and constant conductivity.
- `T2_quench_cpu_kappa` - result-available: CPU quench with Gc(T) and kappa(d) shielding.
- `T2_quench_ref` - result-available: Production half-plate quench matching Jin case 6 (GPU/MPI).

## Convergence and sensitivity studies

- `T2_convective_Bi0p1` - result-available: Convective quench with Bi=0.1 based on Ly/2.
- `T2_convective_Bi10p0` - result-available: Convective quench with Bi=10.0 based on Ly/2.
- `T2_convective_Bi1p0` - result-available: Convective quench with Bi=1.0 based on Ly/2.
- `T2_kres_1em02` - result-available: Residual-conductivity factor design, kappa residual ratio=0.01.
- `T2_kres_1em03` - result-available: Residual-conductivity factor design, kappa residual ratio=0.001.
- `T2_length_l0p25_k0` - configuration-only: Length-scale study, ell=0.25 mm, ell/h=4, constant conductivity.
- `T2_length_l0p25_kd` - configuration-only: Length-scale study, ell=0.25 mm, ell/h=4, degraded conductivity.
- `T2_length_l0p75_k0` - result-available: Length-scale study, ell=0.75 mm, ell/h approximately 4, constant conductivity.
- `T2_length_l0p75_kd` - result-available: Length-scale study, ell=0.75 mm, ell/h approximately 4, degraded conductivity.
- `T2_length_l1p0_k0` - result-available: Length-scale study, ell=1.0 mm, ell/h=4, constant conductivity.
- `T2_length_l1p0_kd` - result-available: Length-scale study, ell=1.0 mm, ell/h=4, degraded conductivity.
- `T2_mesh_050x020_k0` - result-available: Fixed-length mesh-density study: 50x20 QUAD4 cells, ell/h=2.00.
- `T2_mesh_200x080_k0` - result-available: Fixed-length mesh-density study: 200x80 QUAD4 cells, ell/h=8.00.
- `T2_mesh_312x125_k0` - result-available: Fixed-length mesh-density study: 312x125 QUAD4 cells, ell/h=12.48.
- `T2_notched_p0` - result-available: Resolved quench with a 2 mm mid-height edge flaw.
- `T2_p0_at2_k0` - result-available: P0 resolved matched quench: AT2, constant conductivity; ell/h=4.
- `T2_p0_at2_kd` - result-available: P0 resolved matched quench: AT2, damage-dependent conductivity; ell/h=4.
- `T2_reheat_p0` - result-available: Resolved quench-reheat irreversibility control.
- `T2_time_005` - result-available: Fixed-mesh backward-Euler temporal refinement with 5 steps to 0.1 s.
- `T2_time_020` - result-available: Fixed-mesh backward-Euler temporal refinement with 20 steps to 0.1 s.
- `T2_time_020_kd` - result-available: 20-step matched quench with damage-dependent conductivity.
- `T2_time_040` - result-available: Fixed-mesh backward-Euler temporal refinement with 40 steps to 0.1 s.
- `T2_time_040_kd` - result-available: 40-step matched quench with damage-dependent conductivity.
- `T2_time_080` - result-available: Fixed-mesh backward-Euler temporal refinement with 80 steps to 0.1 s.

## Constitutive and optimizer comparisons

- `M1_shear_lbfgs` - result-available: CPU L-BFGS notched shear on the 40x40 Miehe square.
- `M1_shear_lbfgs_ref` - result-available: L-BFGS shear on the 80x80 production mesh, u to 0.020 mm.
- `M1_shear_pfczm` - result-available: Isothermal PF-CZM shear on the Miehe square, monolithic L-BFGS, 80x80.
- `M1_tension_at1_cpu` - result-available: CPU AT1 tension on the 40x40 Miehe square.
- `M1_tension_at1_ref` - result-available: AT1 tension on the 80x80 production mesh.
- `M1_tension_lbfgs` - result-available: CPU L-BFGS notched tension on the 40x40 Miehe square.
- `M1_tension_lbfgs_ref` - result-available: L-BFGS tension on the 80x80 production mesh, u to 0.008 mm.
- `M2_cyclic_lbfgsb` - result-available: CPU L-BFGS-B cyclic shear with box-constraint irreversibility.
- `M4_lshape` - configuration-only: Winkler L-panel, Jin 2024 PF-CZM: bottom fixed, uy at (470, 250).
- `T2_p0_at1_k0` - result-available: P0 resolved matched quench: AT1 strength-threshold surrogate, constant conductivity; ell/h=4.
- `T2_p0_at1_kd` - result-available: P0 resolved matched quench: AT1 strength-threshold surrogate, damage-dependent conductivity; ell/h=4.
- `T2_pfczm_ft0p5_k0` - configuration-only: PF-CZM calibrated-strength comparison, ft=0.5 GPa, constant conductivity.
- `T2_pfczm_ft0p5_kd` - configuration-only: PF-CZM calibrated-strength comparison, ft=0.5 GPa, degraded conductivity.
- `T2_pfczm_p0_k0` - configuration-only: PF-CZM strength-controlled quench, constant conductivity.
- `T2_pfczm_p0_kd` - configuration-only: PF-CZM strength-controlled quench, degraded conductivity.
- `T2_quench_lbfgs_k0` - result-available: CPU L-BFGS quench, kappa0 and constant Gcz.
- `T2_quench_lbfgs_k0gct` - result-available: L-BFGS quench, kappa0 and Gc(T), matching staggered T2_quench_cpu.
- `T2_quench_lbfgs_kd` - result-available: CPU L-BFGS quench, kappa(d) and Gc(T).
- `T2_quench_lbfgs_kd10` - result-available: L-BFGS quench, kappa(d) and Gc(T), matching staggered T2_quench_cpu_kappa.
- `T2_quench_pfczm_lbfgs` - result-available: CPU L-BFGS PF-CZM quench with kappa(d) and Gc(T).
- `T2_quench_pfczm_ref` - result-available: PF-CZM quench on 80x32 (ell=0.2 mm) with kappa(d) and Gc(T).
- `T2_quench_reheat_lbfgsb` - result-available: CPU L-BFGS-B quench then reheat; box constraints, no history field.

## Adaptive and high-density meshes

- `M1_shear_amr` - result-available: AMR shear: damage-driven longest-edge bisection, curved path.
- `M1_shear_cmame` - result-available: CMAME-density AT2 shear: 256×256 QUAD4 to u=0.020 mm.
- `M1_shear_lbfgs_cmame` - result-available: L-BFGS on the 256×256 CMAME shear residual.
- `M1_tension_amr` - result-available: AMR tension: coarse 32x32 TRI3, longest-edge bisection to ℓ/h≥4 in the crack band.
- `M1_tension_amr_cmame` - result-available: AMR tension to ~50k TRI3, target h/ℓ=0.20 (ℓ/h=5) in the band.
- `M1_tension_cmame` - result-available: CMAME-density AT2 tension: 256×256 QUAD4, ℓ/h=3.84, 65536 cells.
- `M1_tension_lbfgs_cmame` - result-available: L-BFGS on the 256×256 CMAME tension residual.
- `T2_amr_p0` - result-available: Conforming TRI3 AMR quench with local ell/h target 4.
- `T2_quench_amr` - result-available: AMR quench: thermal-gradient + damage indicators, κ(d) and Gc(T).
- `T2_quench_amr_cmame` - result-available: AMR quench to ~50k TRI3, thermal-skin target h/ℓ=0.20.
- `T2_quench_cmame_k0` - result-available: CMAME-density quench κ0, Gc(T): 312×125, ℓ/h=5, 39000 cells.
- `T2_quench_cmame_kd` - result-available: CMAME-density quench κ(d), Gc(T): same 312×125 residual as k0.
- `T2_quench_lbfgs_cmame_k0` - result-available: L-BFGS CMAME quench, κ0 and Gc(T), matching T2_quench_cmame_k0.
- `T2_quench_lbfgs_cmame_kd` - result-available: L-BFGS CMAME quench, κ(d) and Gc(T), matching T2_quench_cmame_kd.
- `T2_quench_pfczm_cmame` - result-available: PF-CZM quench on the 312×125 CMAME mesh (ℓ/h=5).

## Three-dimensional thermal shock

- `T3_resolved_path` - result-available: Reduced HEX8 three-dimensional consistency path with ell/h=4.
- `T3_shock_cmame` - result-available: Reduced HEX8 40×16×10 thermal-shock calculation for comparison with the large-scale reference.
- `T3_shock_lbfgs` - result-available: CPU L-BFGS HEX8 thermal shock on a reduced 3D brick (not the 1e7-DOF production mesh).
- `T3_shock_lbfgs_mid` - result-available: Reduced 16×8×5 HEX8 thermal-shock calculation with damage-dependent conductivity.

## Inverse identification

- `I1_gcT_inverse` - result-available: I1: FD outer L-BFGS identification of (Gc0, b1, b2) from a near-edge damage snapshot.
- `I1_gcT_inverse_cpu` - result-available: I1 on the 40x16 T2-CPU mesh: FD outer L-BFGS for (Gc0, b1, b2).
