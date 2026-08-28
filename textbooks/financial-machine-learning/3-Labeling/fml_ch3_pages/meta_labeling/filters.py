"""Filter controls for the meta-labeling page."""

from __future__ import annotations

from dash import html

from maths_self_study.dashboards.components import filter_bar, slider


def build_filters() -> html.Div:
    return filter_bar(
        slider("meta-refresh", "Reload panel", 0, 1, 1, 1),
    )
