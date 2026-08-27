"""Filter controls for the KKT conditions page."""

from __future__ import annotations

from dash import html

from maths_self_study.dashboards.components import filter_bar, slider
from maths_self_study.demos.deep_learning import ch4 as helpers


def build_filters() -> html.Div:
    return filter_bar(
        slider(
            "kkt-bound",
            "Constraint lower bound b (aᵀx ≥ b)",
            -0.5,
            2.5,
            helpers.KKT_LOWER_BOUND,
            step=0.1,
        ),
    )
