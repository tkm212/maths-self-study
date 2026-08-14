"""Filter controls for the distributions page."""

from __future__ import annotations

from dash import html

from maths_self_study.dashboards.components import filter_bar, num_input, section
from maths_self_study.deep_learning import ch3_helpers as helpers


def build_filters() -> html.Div:
    c = helpers.CATEGORICAL_PROBS
    cov = helpers.GAUSSIAN_2D_COV
    return filter_bar(
        section(
            "Categorical probs",
            num_input("dist-c0", "P(A)", float(c[0])),
            num_input("dist-c1", "P(B)", float(c[1])),
            num_input("dist-c2", "P(C)", float(c[2])),
            num_input("dist-c3", "P(D)", float(c[3])),
        ),
        section(
            "2D Gaussian covariance",
            num_input("dist-cov11", "Σ₁₁", float(cov[0, 0]), step=0.1),
            num_input("dist-cov12", "Σ₁₂=Σ₂₁", float(cov[0, 1]), step=0.1),
            num_input("dist-cov22", "Σ₂₂", float(cov[1, 1]), step=0.1),
        ),
    )
