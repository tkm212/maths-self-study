"""Filter controls."""

from __future__ import annotations

from dash import html

from maths_self_study.dashboards.components import filter_bar, slider


def build_filters() -> html.Div:
    return filter_bar(
        slider("ppr-ridge-m", "Ridge terms M (visualisation)", 1, 10, 3, 1),
    )
