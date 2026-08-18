"""Filter controls for the Newton page."""

from __future__ import annotations

from dash import html

from maths_self_study.dashboards.components import filter_bar, num_input, slider
from maths_self_study.deep_learning import ch4_helpers as helpers


def build_filters() -> html.Div:
    start = helpers.NEWTON_START
    return filter_bar(
        slider("newton-eta", "GD learning rate η", 0.01, 0.3, 0.08, step=0.01),
        num_input("newton-x0", "Start x₁", float(start[0]), step=0.1),
        num_input("newton-x1", "Start x₂", float(start[1]), step=0.1),
    )
