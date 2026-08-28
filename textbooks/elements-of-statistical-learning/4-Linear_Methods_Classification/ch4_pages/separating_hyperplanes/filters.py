"""Filter controls for separating hyperplanes page."""

from __future__ import annotations

from dash import html

from maths_self_study.dashboards.components import filter_bar, num_input, slider


def build_filters() -> html.Div:
    return filter_bar(
        num_input("sh-n", "Sample size n", 200, step=50, min_=50),
        slider("sh-margin", "Class margin", 0.5, 3.0, 1.5, 0.1),
        num_input("sh-seed", "Random seed", 42, step=1),
    )
