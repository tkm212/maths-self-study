"""Filter controls for the vectors page."""

from __future__ import annotations

from dash import html

from maths_self_study.dashboards.components import filter_bar, matrix_input, slider
from maths_self_study.demos.deep_learning import ch2 as helpers


def build_filters() -> html.Div:
    return filter_bar(
        matrix_input("grid-matrix", "Grid map A", helpers.GRID_MAP),
        slider("vm-rot", "Rotation (°)", -180, 180, 30, 1),
        slider("vm-shear", "Shear k", -2.0, 2.0, 0.8, 0.1),
    )
