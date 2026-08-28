"""Filter controls for the distributions page."""

from __future__ import annotations

from dash import html

from maths_self_study.dashboards.components import filter_bar, matrix_input, num_input, section
from maths_self_study.demos.deep_learning import ch3 as helpers


def build_filters() -> html.Div:
    c = helpers.CATEGORICAL_PROBS
    return filter_bar(
        section(
            "Categorical probs",
            num_input("dist-c0", "P(A)", float(c[0])),
            num_input("dist-c1", "P(B)", float(c[1])),
            num_input("dist-c2", "P(C)", float(c[2])),
            num_input("dist-c3", "P(D)", float(c[3])),
        ),
        section(
            "2D Gaussian covariance Σ",
            matrix_input("dist-cov-matrix", "Σ", helpers.GAUSSIAN_2D_COV),
        ),
    )
