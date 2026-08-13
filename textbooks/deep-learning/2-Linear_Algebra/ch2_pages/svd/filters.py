"""Filter controls for the SVD page."""

from __future__ import annotations

from dash import html

from maths_self_study.dashboards.components import filter_bar, matrix_inputs, num_input
from maths_self_study.deep_learning import ch2_helpers as helpers


def build_filters() -> html.Div:
    return filter_bar(
        matrix_inputs("svd", helpers.SVD_MAP, "Map A"),
        html.Div("Least-squares b", style={"fontWeight": 600, "width": "100%"}),
        num_input("svd-b0", "b₀", float(helpers.OVERDETERMINED_B[0]), step=0.5),
        num_input("svd-b1", "b₁", float(helpers.OVERDETERMINED_B[1]), step=0.5),
        num_input("svd-b2", "b₂", float(helpers.OVERDETERMINED_B[2]), step=0.5),
    )
