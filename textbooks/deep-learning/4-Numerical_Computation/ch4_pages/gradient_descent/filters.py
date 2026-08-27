"""Filter controls for the gradient descent page."""

from __future__ import annotations

from dash import html

from maths_self_study.dashboards.components import filter_bar, slider, vector2_input
from maths_self_study.demos.deep_learning import ch4 as helpers


def build_filters() -> html.Div:
    return filter_bar(
        slider("gd-eta", "Learning rate η", 0.01, 0.5, 0.1, step=0.01),
        vector2_input("gd", helpers.GD_START, labels=("Start x₁", "Start x₂")),
    )
