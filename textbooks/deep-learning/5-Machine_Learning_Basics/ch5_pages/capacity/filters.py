"""Filter controls for the capacity page."""

from __future__ import annotations

from dash import html

from maths_self_study.dashboards.components import filter_bar, slider
from maths_self_study.deep_learning import ch5_helpers as helpers


def build_filters() -> html.Div:
    return filter_bar(
        slider("cap-degree", "Polynomial degree", 1, 12, helpers.CAPACITY_DEGREE, step=1),
        slider("cap-noise", "Label noise sigma", 0.0, 0.4, helpers.CAPACITY_NOISE, step=0.02),
    )
