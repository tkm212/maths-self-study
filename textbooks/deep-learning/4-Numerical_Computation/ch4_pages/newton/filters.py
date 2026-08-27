"""Filter controls for the Newton page."""

from __future__ import annotations

from dash import html

from maths_self_study.dashboards.components import filter_bar, slider, vector2_input
from maths_self_study.deep_learning import ch4_helpers as helpers


def build_filters() -> html.Div:
    return filter_bar(
        slider("newton-eta", "GD learning rate η", 0.01, 0.3, 0.08, step=0.01),
        vector2_input("newton", helpers.NEWTON_START, labels=("Start x₁", "Start x₂")),
    )
