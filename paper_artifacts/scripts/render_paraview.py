#!/usr/bin/env python3
"""Off-screen ParaView screenshots of executed JAX-TPFF VTU fields."""

from __future__ import annotations

import os
import sys

from paraview.simple import (
    XMLUnstructuredGridReader,
    GetActiveViewOrCreate,
    Show,
    Hide,
    HideAll,
    ColorBy,
    GetColorTransferFunction,
    GetOpacityTransferFunction,
    GetScalarBar,
    HideScalarBarIfNotNeeded,
    Render,
    ResetCamera,
    SaveScreenshot,
    Delete,
    LoadPalette,
    GetSources,
)

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
OUT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "figures"))
os.makedirs(OUT, exist_ok=True)

# Damage: pale → amber → crimson (standard fracture-paper palette).
D_RGB = [
    0.00, 0.97, 0.97, 0.96,
    0.20, 0.99, 0.90, 0.55,
    0.50, 0.96, 0.50, 0.12,
    0.80, 0.78, 0.08, 0.12,
    1.00, 0.28, 0.00, 0.08,
]
# Temperature: perceptually uniform viridis sequence, cool → hot (K).
T_RGB = [
    293.15, 0.267004, 0.004874, 0.329415,
    363.15, 0.229739, 0.322361, 0.545706,
    433.15, 0.127568, 0.566949, 0.550556,
    503.15, 0.369214, 0.788888, 0.382914,
    573.15, 0.993248, 0.906157, 0.143936,
]


def _lut(name, rgb, rng):
    lut = GetColorTransferFunction(name)
    lut.RGBPoints = list(rgb)
    lut.ColorSpace = "RGB"
    lut.RescaleTransferFunction(*rng)
    lut.NanColor = [0.5, 0.5, 0.5]
    GetOpacityTransferFunction(name).RescaleTransferFunction(*rng)
    return lut


def _bar(lut, view, title, profile="half"):
    # The PNGs are reduced substantially in the review-format manuscript.
    # Scale type explicitly by final panel width so that labels remain at
    # journal-readable size after LaTeX placement.
    sizes = {
        "third": (96, 82, 34),
        "half": (76, 64, 30),
        "single": (64, 54, 28),
    }
    title_size, label_size, thickness = sizes.get(profile, sizes["half"])
    bar = GetScalarBar(lut, view)
    bar.Title = title
    bar.ComponentTitle = ""
    bar.TitleFontFamily = "Times"
    bar.LabelFontFamily = "Times"
    bar.TitleFontSize = title_size
    bar.LabelFontSize = label_size
    bar.TitleColor = [0, 0, 0]
    bar.LabelColor = [0, 0, 0]
    bar.ScalarBarThickness = thickness
    bar.ScalarBarLength = 0.68
    bar.WindowLocation = "Any Location"
    bar.Position = [0.885, 0.16]
    bar.AutomaticLabelFormat = 0
    bar.AddRangeLabels = 0
    bar.DrawTickLabels = 1
    if title == "d":
        bar.LabelFormat = "{:.1f}"
        bar.UseCustomLabels = 1
        bar.CustomLabels = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
    else:
        bar.LabelFormat = "{:.0f}"
        bar.UseCustomLabels = 1
        bar.CustomLabels = [293.15, 350.0, 400.0, 450.0, 500.0, 573.15]
    return bar


def _reset_view(view_size):
    HideAll()
    for src in list(GetSources().values()):
        try:
            Delete(src)
        except Exception:
            pass
    view = GetActiveViewOrCreate("RenderView")
    view.ViewSize = list(view_size)
    view.OrientationAxesVisibility = 0
    view.Background = [1, 1, 1]
    view.UseColorPaletteForBackground = 0
    for name in ("d", "T", "u", "sol", "history"):
        try:
            lut = GetColorTransferFunction(name)
            bar = GetScalarBar(lut, view)
            bar.Visibility = 0
            HideScalarBarIfNotNeeded(lut, view)
        except Exception:
            pass
    try:
        LoadPalette("WhiteBackground")
    except Exception:
        pass
    return view


def render_field(vtu, array, out_png, rng, rgb, title, mode="2D", view_size=(2400, 1000)):
    if not os.path.isfile(vtu):
        print("MISSING", vtu)
        return False
    view = _reset_view(view_size)
    reader = XMLUnstructuredGridReader(FileName=[vtu])
    reader.PointArrayStatus = [array]
    reader.UpdatePipeline()

    disp = Show(reader, view)
    disp.Representation = "Surface"
    disp.EdgeColor = [0.12, 0.12, 0.12]
    if ("amr" in vtu and "cmame" not in vtu) or "edges" in mode:
        disp.Representation = "Surface With Edges"
        disp.LineWidth = 0.45
        try:
            disp.EdgeOpacity = 0.48
        except Exception:
            pass
    ColorBy(disp, ("POINTS", array))
    lut = _lut(array, rgb, rng)
    lut.RescaleTransferFunction(*rng)
    for name in ("d", "T", "u", "sol", "history"):
        if name == array:
            continue
        try:
            GetScalarBar(GetColorTransferFunction(name), view).Visibility = 0
        except Exception:
            pass
    profile = "third" if "third" in mode else "single" if "single" in mode else "half"
    bar = _bar(lut, view, title, profile=profile)
    if mode.startswith("3D"):
        bar.Position = [0.885, 0.16]
    bar.Visibility = 1
    disp.SetScalarBarVisibility(view, True)

    if mode.startswith("2D"):
        view.InteractionMode = "2D"
        view.CameraParallelProjection = 1
        ResetCamera(view)
        cam = view.GetActiveCamera()
        fp = cam.GetFocalPoint()
        cam.SetPosition(fp[0], fp[1], fp[2] + 20.0)
        cam.SetViewUp(0, 1, 0)
        ResetCamera(view)
        view.CameraParallelProjection = 1
        # ResetCamera leaves space for the scalar bar and is deliberately
        # conservative.  Tighten the framing so that the plate occupies most
        # of a journal panel while retaining the complete boundary.
        view.CameraParallelScale = 0.47 * float(view.CameraParallelScale)
    else:
        view.InteractionMode = "3D"
        view.CameraParallelProjection = 0
        ResetCamera(view)
        cam = view.GetActiveCamera()
        fp = list(cam.GetFocalPoint())
        cam.SetFocalPoint(fp[0], fp[1], fp[2])
        cam.SetPosition(fp[0] + 16.0, fp[1] - 14.0, fp[2] + 9.0)
        cam.SetViewUp(0, 0, 1)
        ResetCamera(view)
        cam.Azimuth(-22)
        cam.Elevation(16)
        cam.Zoom(1.05)

    Render(view)
    SaveScreenshot(
        out_png,
        view,
        ImageResolution=list(view_size),
        CompressionLevel=1,
        TransparentBackground=0,
    )
    print("wrote", out_png)
    try:
        disp.SetScalarBarVisibility(view, False)
        bar.Visibility = 0
        Hide(reader, view)
        Delete(disp)
        Delete(reader)
    except Exception:
        pass
    return True


JOBS = [
    ("results/manuscript/TS_200x080_n080_k0/vtk/step_0080.vtu", "d", "manuscript_main_k0_d.png", (0, 1), D_RGB, "d", "2D-half"),
    ("results/manuscript/TS_200x080_n080_k0/vtk/step_0080.vtu", "T", "manuscript_main_k0_T.png", (293.15, 573.15), T_RGB, "T [K]", "2D-half"),
    ("results/manuscript/TS_200x080_n080_kd/vtk/step_0080.vtu", "d", "manuscript_main_kd_d.png", (0, 1), D_RGB, "d", "2D-half"),
    ("results/manuscript/TS_200x080_n080_kd/vtk/step_0080.vtu", "T", "manuscript_main_kd_T.png", (293.15, 573.15), T_RGB, "T [K]", "2D-half"),
    ("results/manuscript/TS_100x040_n040_kd/vtk/step_0040.vtu", "d", "manuscript_mesh_100_kd_d.png", (0, 1), D_RGB, "d", "2D-third-edges"),
    ("results/manuscript/TS_200x080_n040_kd/vtk/step_0040.vtu", "d", "manuscript_mesh_200_kd_d.png", (0, 1), D_RGB, "d", "2D-third-edges"),
    ("results/manuscript/TS_400x160_n040_kd/vtk/step_0040.vtu", "d", "manuscript_mesh_400_kd_d.png", (0, 1), D_RGB, "d", "2D-third-edges"),
    ("results/manuscript/TS_Bi10_n080_k0/vtk/step_0080.vtu", "T", "manuscript_bi10_k0_T.png", (293.15, 573.15), T_RGB, "T [K]", "2D-half"),
    ("results/manuscript/TS_Bi10_n080_kd/vtk/step_0080.vtu", "T", "manuscript_bi10_kd_T.png", (293.15, 573.15), T_RGB, "T [K]", "2D-half"),
    ("results/manuscript/TS_free_n080_k0/vtk/step_0080.vtu", "d", "manuscript_free_k0_d.png", (0, 1), D_RGB, "d", "2D-half"),
    ("results/manuscript/TS_free_n080_kd/vtk/step_0080.vtu", "d", "manuscript_free_kd_d.png", (0, 1), D_RGB, "d", "2D-half"),
    ("results/manuscript/TS_AT1_n080_k0/vtk/step_0080.vtu", "d", "manuscript_at1_k0_d.png", (0, 1), D_RGB, "d", "2D-half"),
    ("results/manuscript/TS_AT1_n080_kd/vtk/step_0080.vtu", "d", "manuscript_at1_kd_d.png", (0, 1), D_RGB, "d", "2D-half"),
    ("results/verification/T2_time_020/vtk/step_0020.vtu", "d", "val_t2_time020_k0_d.png", (0, 1), D_RGB, "d", "2D-half"),
    ("results/verification/T2_time_020/vtk/step_0020.vtu", "T", "val_t2_time020_k0_T.png", (293.15, 573.15), T_RGB, "T [K]", "2D-half"),
    ("results/verification/T2_time_020_kd/vtk/step_0020.vtu", "d", "val_t2_time020_kd_d.png", (0, 1), D_RGB, "d", "2D-half"),
    ("results/verification/T2_time_020_kd/vtk/step_0020.vtu", "T", "val_t2_time020_kd_T.png", (293.15, 573.15), T_RGB, "T [K]", "2D-half"),
    ("results/verification/T2_time_080/vtk/step_0080.vtu", "d", "val_t2_time080_k0_d.png", (0, 1), D_RGB, "d", "2D-edges"),
    ("outputs_expansion/T2_mesh_050x020_k0/vtk/step_0010.vtu", "d", "exp_mesh_050x020_d.png", (0, 1), D_RGB, "d", "2D-half-edges"),
    ("outputs_p0/T2_p0_at2_k0/vtk/step_0010.vtu", "d", "exp_mesh_100x040_d.png", (0, 1), D_RGB, "d", "2D-half-edges"),
    ("outputs_expansion/T2_mesh_200x080_k0/vtk/step_0010.vtu", "d", "exp_mesh_200x080_d.png", (0, 1), D_RGB, "d", "2D-half-edges"),
    ("outputs_expansion/T2_mesh_312x125_k0/vtk/step_0010.vtu", "d", "exp_mesh_312x125_d.png", (0, 1), D_RGB, "d", "2D-half-edges"),
    ("outputs_expansion/T2_convective_Bi0p1/vtk/step_0010.vtu", "T", "exp_conv_Bi01_T.png", (293.15, 573.15), T_RGB, "T [K]", "2D-third"),
    ("outputs_expansion/T2_convective_Bi1p0/vtk/step_0010.vtu", "T", "exp_conv_Bi1_T.png", (293.15, 573.15), T_RGB, "T [K]", "2D-third"),
    ("outputs_expansion/T2_convective_Bi1p0/vtk/step_0010.vtu", "d", "exp_conv_Bi1_d.png", (0, 1), D_RGB, "d", "2D-third"),
    ("outputs_expansion/T2_notched_p0/vtk/step_0010.vtu", "d", "exp_notched_d.png", (0, 1), D_RGB, "d", "2D-single"),
    ("outputs_expansion/T2_reheat_p0/vtk/step_0005.vtu", "T", "exp_reheat_quench_T.png", (293.15, 573.15), T_RGB, "T [K]", "2D-third"),
    ("outputs_expansion/T2_reheat_p0/vtk/step_0010.vtu", "T", "exp_reheat_final_T.png", (293.15, 573.15), T_RGB, "T [K]", "2D-third"),
    ("outputs_expansion/T2_reheat_p0/vtk/step_0010.vtu", "d", "exp_reheat_final_d.png", (0, 1), D_RGB, "d", "2D-third"),
    ("outputs_expansion/T2_amr_p0/vtk/step_0010.vtu", "d", "exp_amr_d.png", (0, 1), D_RGB, "d", "2D-single-edges"),
    ("outputs_expansion/T3_resolved_path/vtk/step_0005.vtu", "d", "exp_t3_d.png", (0, 1), D_RGB, "d", "3D-half"),
    ("outputs_expansion/T3_resolved_path/vtk/step_0005.vtu", "T", "exp_t3_T.png", (293.15, 573.15), T_RGB, "T [K]", "3D-half"),
    ("outputs_expansion/T2_length_l1p0_k0/vtk/step_0010.vtu", "d", "exp_length_l1_d.png", (0, 1), D_RGB, "d", "2D"),
    ("outputs_p0/T2_p0_at2_k0/vtk/step_0010.vtu", "d", "t2_p0_at2_k0_d.png", (0, 1), D_RGB, "d", "2D"),
    ("outputs_p0/T2_p0_at2_kd/vtk/step_0010.vtu", "d", "t2_p0_at2_kd_d.png", (0, 1), D_RGB, "d", "2D"),
    ("outputs_p0/T2_p0_at2_k0/vtk/step_0010.vtu", "T", "t2_p0_at2_k0_T.png", (293.15, 573.15), T_RGB, "T [K]", "2D"),
    ("outputs_p0/T2_p0_at2_kd/vtk/step_0010.vtu", "T", "t2_p0_at2_kd_T.png", (293.15, 573.15), T_RGB, "T [K]", "2D"),
    ("outputs_p0/T2_p0_at1_k0/vtk/step_0010.vtu", "d", "t2_p0_at1_k0_d.png", (0, 1), D_RGB, "d", "2D"),
    ("outputs/M1_tension_ref/vtk/step_0039.vtu", "d", "m1_tension_ref_d.png", (0, 1), D_RGB, "d", "2D"),
    ("outputs/M1_tension_lbfgs/vtk/step_0016.vtu", "d", "m1_tension_lbfgs_d.png", (0, 1), D_RGB, "d", "2D"),
    ("outputs/M1_tension_cpu/vtk/step_0031.vtu", "d", "m1_tension_cpu_d.png", (0, 1), D_RGB, "d", "2D"),
    ("outputs/M1_tension_at1_cpu/vtk/step_0031.vtu", "d", "m1_tension_at1_d.png", (0, 1), D_RGB, "d", "2D"),
    ("outputs/M1_tension_amr/vtk/step_0037.vtu", "d", "m1_tension_amr_d.png", (0, 1), D_RGB, "d", "2D"),
    ("outputs/M1_shear_ref/vtk/step_0039.vtu", "d", "m1_shear_ref_d.png", (0, 1), D_RGB, "d", "2D"),
    ("outputs/M1_shear_lbfgs/vtk/step_0016.vtu", "d", "m1_shear_lbfgs_d.png", (0, 1), D_RGB, "d", "2D"),
    ("outputs/M1_shear_amr/vtk/step_0033.vtu", "d", "m1_shear_amr_d.png", (0, 1), D_RGB, "d", "2D"),
    ("outputs/M2_cyclic_lbfgsb/vtk/step_0012.vtu", "d", "m2_unload_d.png", (0, 1), D_RGB, "d", "2D"),
    ("outputs/M2_cyclic_lbfgsb/vtk/step_0022.vtu", "d", "m2_reload_d.png", (0, 1), D_RGB, "d", "2D"),
    ("outputs/T1_thermo_shear_cpu/vtk/step_0024.vtu", "d", "t1_d.png", (0, 1), D_RGB, "d", "2D"),
    ("outputs/T1_thermo_shear_cpu/vtk/step_0024.vtu", "T", "t1_T.png", (293.15, 573.15), T_RGB, "T [K]", "2D"),
    ("outputs/T2_quench_cpu/vtk/step_0020.vtu", "d", "t2_k0_d.png", (0, 1), D_RGB, "d", "2D"),
    ("outputs/T2_quench_cpu/vtk/step_0020.vtu", "T", "t2_k0_T.png", (293.15, 573.15), T_RGB, "T [K]", "2D"),
    ("outputs/T2_quench_cpu_kappa/vtk/step_0020.vtu", "d", "t2_kd_d.png", (0, 1), D_RGB, "d", "2D"),
    ("outputs/T2_quench_cpu_kappa/vtk/step_0020.vtu", "T", "t2_kd_T.png", (293.15, 573.15), T_RGB, "T [K]", "2D"),
    ("outputs/T2_quench_lbfgs_k0/vtk/step_0012.vtu", "d", "t2_lbfgs_k0_d.png", (0, 1), D_RGB, "d", "2D"),
    ("outputs/T2_quench_lbfgs_kd/vtk/step_0012.vtu", "d", "t2_lbfgs_kd_d.png", (0, 1), D_RGB, "d", "2D"),
    ("outputs/T2_quench_pfczm_lbfgs/vtk/step_0012.vtu", "d", "t2_pfczm_d.png", (0, 1), D_RGB, "d", "2D"),
    ("outputs/T2_quench_reheat_lbfgsb/vtk/step_0016.vtu", "d", "t2_reheat_d.png", (0, 1), D_RGB, "d", "2D"),
    ("outputs/T2_quench_reheat_lbfgsb/vtk/step_0016.vtu", "T", "t2_reheat_T.png", (293.15, 573.15), T_RGB, "T [K]", "2D"),
    ("outputs/T2_quench_ref/vtk/step_0200.vtu", "d", "t2_ref_d.png", (0, 1), D_RGB, "d", "2D"),
    ("outputs/T2_quench_ref/vtk/step_0200.vtu", "T", "t2_ref_T.png", (293.15, 573.15), T_RGB, "T [K]", "2D"),
    ("outputs/T2_quench_amr/vtk/step_0024.vtu", "d", "t2_amr_d.png", (0, 1), D_RGB, "d", "2D"),
    ("outputs/T3_shock_lbfgs/vtk/step_0006.vtu", "d", "t3_d.png", (0, 1), D_RGB, "d", "3D"),
    ("outputs/I1_gcT_inverse/obs/vtk/step_0008.vtu", "d", "i1_obs_d.png", (0, 1), D_RGB, "d", "2D"),
    ("outputs/I1_gcT_inverse/fwd/vtk/step_0008.vtu", "d", "i1_hat_d.png", (0, 1), D_RGB, "d", "2D"),
    ("outputs/M1_tension_lbfgs_ref/vtk/step_0033.vtu", "d", "m1_tension_lbfgs_ref_d.png", (0, 1), D_RGB, "d", "2D"),
    ("outputs/M1_shear_lbfgs_ref/vtk/step_0033.vtu", "d", "m1_shear_lbfgs_ref_d.png", (0, 1), D_RGB, "d", "2D"),
    ("outputs/M1_shear_ref_ext/vtk/step_0041.vtu", "d", "m1_shear_ref_ext_d.png", (0, 1), D_RGB, "d", "2D"),
    ("outputs/T1_thermo_shear_ref/vtk/step_0033.vtu", "d", "t1_ref_d.png", (0, 1), D_RGB, "d", "2D"),
    ("outputs/T1_thermo_shear_ref/vtk/step_0033.vtu", "T", "t1_ref_T.png", (293.15, 573.15), T_RGB, "T [K]", "2D"),
    ("outputs/T2_quench_lbfgs_k0gct/vtk/step_0020.vtu", "d", "t2_lbfgs_k0gct_d.png", (0, 1), D_RGB, "d", "2D"),
    ("outputs/T2_quench_lbfgs_k0gct/vtk/step_0020.vtu", "T", "t2_lbfgs_k0gct_T.png", (293.15, 573.15), T_RGB, "T [K]", "2D"),
    ("outputs/T2_quench_lbfgs_kd10/vtk/step_0020.vtu", "d", "t2_lbfgs_kd10_d.png", (0, 1), D_RGB, "d", "2D"),
    ("outputs/T2_quench_lbfgs_kd10/vtk/step_0020.vtu", "T", "t2_lbfgs_kd10_T.png", (293.15, 573.15), T_RGB, "T [K]", "2D"),
    ("outputs/T2_quench_pfczm_ref/vtk/step_0020.vtu", "d", "t2_pfczm_ref_d.png", (0, 1), D_RGB, "d", "2D"),
    ("outputs/T3_shock_lbfgs_mid/vtk/step_0012.vtu", "d", "t3_mid_d.png", (0, 1), D_RGB, "d", "3D"),
    ("outputs/I1_gcT_inverse_cpu/obs/vtk/step_0016.vtu", "d", "i1_cpu_obs_d.png", (0, 1), D_RGB, "d", "2D"),
    ("outputs/I1_gcT_inverse_cpu/fwd/vtk/step_0016.vtu", "d", "i1_cpu_hat_d.png", (0, 1), D_RGB, "d", "2D"),
    ("outputs/M1_tension_cmame/vtk/step_0033.vtu", "d", "m1_tension_cmame_d.png", (0, 1), D_RGB, "d", "2D"),
    ("outputs/M1_shear_cmame/vtk/step_0033.vtu", "d", "m1_shear_cmame_d.png", (0, 1), D_RGB, "d", "2D"),
    ("outputs/T2_quench_cmame_k0/vtk/step_0020.vtu", "d", "t2_cmame_k0_d.png", (0, 1), D_RGB, "d", "2D"),
    ("outputs/T2_quench_cmame_k0/vtk/step_0020.vtu", "T", "t2_cmame_k0_T.png", (293.15, 573.15), T_RGB, "T [K]", "2D"),
    ("outputs/T2_quench_cmame_kd/vtk/step_0020.vtu", "d", "t2_cmame_kd_d.png", (0, 1), D_RGB, "d", "2D"),
    ("outputs/T2_quench_cmame_kd/vtk/step_0020.vtu", "T", "t2_cmame_kd_T.png", (293.15, 573.15), T_RGB, "T [K]", "2D"),
    ("outputs/T2_quench_pfczm_cmame/vtk/step_0020.vtu", "d", "t2_pfczm_cmame_d.png", (0, 1), D_RGB, "d", "2D"),
    ("outputs/M1_tension_amr_cmame/vtk/step_0033.vtu", "d", "m1_tension_amr_cmame_d.png", (0, 1), D_RGB, "d", "2D"),
    ("outputs/T2_quench_amr_cmame/vtk/step_0020.vtu", "d", "t2_amr_cmame_d.png", (0, 1), D_RGB, "d", "2D"),
    ("outputs/T2_quench_amr_cmame/vtk/step_0020.vtu", "T", "t2_amr_cmame_T.png", (293.15, 573.15), T_RGB, "T [K]", "2D"),
    ("outputs/T3_shock_cmame/vtk/step_0012.vtu", "d", "t3_cmame_d.png", (0, 1), D_RGB, "d", "3D"),
    ("outputs/M3_tpb/vtk/step_0020.vtu", "d", "m3_tpb_d.png", (0, 1), D_RGB, "d", "2D"),
    ("outputs/M5_hole/vtk/step_0020.vtu", "d", "m5_hole_d.png", (0, 1), D_RGB, "d", "2D"),
    ("outputs/M1_shear_pfczm/vtk/step_0019.vtu", "d", "m1_shear_pfczm_d.png", (0, 1), D_RGB, "d", "2D"),
    ("outputs/M4_lshape/vtk/step_0004.vtu", "d", "m4_lshape_d.png", (0, 1), D_RGB, "d", "2D"),
    ("outputs/M1_tension_lbfgs_cmame/vtk/step_0033.vtu", "d", "m1_tension_lbfgs_cmame_d.png", (0, 1), D_RGB, "d", "2D"),
    ("outputs/M1_shear_lbfgs_cmame/vtk/step_0033.vtu", "d", "m1_shear_lbfgs_cmame_d.png", (0, 1), D_RGB, "d", "2D"),
    ("outputs/T2_quench_lbfgs_cmame_k0/vtk/step_0020.vtu", "d", "t2_lbfgs_cmame_k0_d.png", (0, 1), D_RGB, "d", "2D"),
    ("outputs/T2_quench_lbfgs_cmame_kd/vtk/step_0020.vtu", "d", "t2_lbfgs_cmame_kd_d.png", (0, 1), D_RGB, "d", "2D"),
    ("outputs/T3_shock_lbfgs_mid/vtk/step_0012.vtu", "T", "t3_mid_T.png", (293.15, 573.15), T_RGB, "T [K]", "3D"),
    ("outputs/T3_shock_cmame/vtk/step_0012.vtu", "T", "t3_cmame_T.png", (293.15, 573.15), T_RGB, "T [K]", "3D"),
]


def main():
    want = set(sys.argv[1:])
    n_ok = 0
    jobs = [j for j in JOBS if not want or j[2] in want]
    for rel, array, name, rng, rgb, title, mode in jobs:
        vtu = os.path.join(ROOT, rel)
        out = os.path.join(OUT, name)
        vs = (3000, 1100) if mode.startswith("2D") else (3000, 2100)
        try:
            ok = render_field(vtu, array, out, rng, rgb, title, mode=mode, view_size=vs)
        except Exception as exc:
            print("FAIL", name, type(exc).__name__, exc)
            ok = False
        if ok:
            n_ok += 1
    print("done", n_ok, "/", len(jobs))
    return 0 if n_ok == len(jobs) else 1


if __name__ == "__main__":
    sys.exit(main())
