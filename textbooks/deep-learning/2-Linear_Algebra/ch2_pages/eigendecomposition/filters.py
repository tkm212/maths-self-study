"""Filter controls for the eigendecomposition page."""

from __future__ import annotations

from dash import html

from maths_self_study.dashboards.components import filter_bar, matrix_input
from maths_self_study.deep_learning import ch2_helpers as helpers


def build_filters() -> html.Div:
    return filter_bar(
        matrix_input(
            "cov-matrix",
            "Matrix A",
            helpers.COV_2X2,
            hint="Symmetric part (A + Aᵀ)/2 is used for eigh.",
        ),
    )
