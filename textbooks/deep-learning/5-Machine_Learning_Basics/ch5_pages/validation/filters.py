"""Filter controls for the validation page."""

from __future__ import annotations

from dash import html

from maths_self_study.dashboards.components import filter_bar, slider
from maths_self_study.demos.deep_learning import ch5 as helpers


def build_filters() -> html.Div:
    return filter_bar(
        slider("val-l2", "Ridge penalty lambda", 1e-4, 100.0, helpers.VALIDATION_L2, step=0.01),
    )
