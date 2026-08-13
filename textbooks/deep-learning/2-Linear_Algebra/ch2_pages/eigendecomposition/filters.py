"""Filter controls for the eigendecomposition page."""

from __future__ import annotations

from dash import html

from maths_self_study.dashboards.components import filter_bar, matrix_inputs
from maths_self_study.deep_learning import ch2_helpers as helpers


def build_filters() -> html.Div:
    return filter_bar(matrix_inputs("cov", helpers.COV_2X2, "Matrix A (symmetrised for eigh)"))
