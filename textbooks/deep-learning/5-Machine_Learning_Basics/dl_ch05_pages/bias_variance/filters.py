"""Filter controls for the bias-variance page."""

from __future__ import annotations

from dash import html

from maths_self_study.dashboards.components import filter_bar, slider


def build_filters() -> html.Div:
    return filter_bar(
        slider("bv-degree", "Highlight degree", 1, 12, 6, step=1),
    )
