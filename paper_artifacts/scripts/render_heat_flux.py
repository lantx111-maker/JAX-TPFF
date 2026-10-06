#!/usr/bin/env python3
"""Render terminal heat-flux magnitude fields for the IJHMT manuscript."""

from __future__ import annotations

import os

from paraview.simple import (
    Calculator,
    ColorBy,
    Delete,
    GetActiveViewOrCreate,
    GetColorTransferFunction,
    GetOpacityTransferFunction,
    Gradient,
    Hide,
    Render,
    ResetCamera,
    SaveScreenshot,
    Show,
    XMLUnstructuredGridReader,
)

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
FIG = os.path.join(ROOT, "paper_artifacts", "figures")
os.makedirs(FIG, exist_ok=True)


def render(case: str, law: str) -> None:
    source = os.path.join(ROOT, "results/manuscript", case, "vtk", "step_0080.vtu")
    target = os.path.join(FIG, f"manuscript_main_{law}_q.png")
    reader = XMLUnstructuredGridReader(FileName=[source])
    reader.PointArrayStatus = ["T", "d"]
    gradient = Gradient(Input=reader)
    gradient.ScalarArray = ["POINTS", "T"]
    gradient.ResultArrayName = "gradT"
    flux = Calculator(Input=gradient)
    flux.ResultArrayName = "qmag"
    if law == "k0":
        flux.Function = "0.031*mag(gradT)"
    else:
        flux.Function = "0.031*(0.01+0.99*(1-d)^2)*mag(gradT)"
    flux.UpdatePipeline()

    view = GetActiveViewOrCreate("RenderView")
    view.ViewSize = [3000, 1200]
    view.OrientationAxesVisibility = 0
    view.Background = [1.0, 1.0, 1.0]
    view.InteractionMode = "2D"
    view.CameraParallelProjection = 1
    display = Show(flux, view)
    display.Representation = "Surface"
    ColorBy(display, ("POINTS", "qmag"))
    lut = GetColorTransferFunction("qmag")
    lut.RGBPoints = [
        0.0, 0.001462, 0.000466, 0.013866,
        1.3125, 0.341500, 0.062325, 0.429425,
        2.6250, 0.735683, 0.215906, 0.330245,
        3.9375, 0.978422, 0.557937, 0.034931,
        5.2500, 0.988362, 0.998364, 0.644924,
    ]
    lut.ColorSpace = "RGB"
    lut.RescaleTransferFunction(0.0, 5.25)
    GetOpacityTransferFunction("qmag").RescaleTransferFunction(0.0, 5.25)
    display.SetScalarBarVisibility(view, False)

    ResetCamera(view)
    camera = view.GetActiveCamera()
    focus = camera.GetFocalPoint()
    camera.SetPosition(focus[0], focus[1], focus[2] + 20.0)
    camera.SetViewUp(0, 1, 0)
    ResetCamera(view)
    view.CameraParallelProjection = 1
    view.CameraParallelScale = 0.47 * float(view.CameraParallelScale)
    Render(view)
    SaveScreenshot(target, view, ImageResolution=[3000, 1200], CompressionLevel=1)
    print("wrote", target)
    Hide(flux, view)
    Delete(display)
    Delete(flux)
    Delete(gradient)
    Delete(reader)


render("TS_200x080_n080_k0", "k0")
render("TS_200x080_n080_kd", "kd")
