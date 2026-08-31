"""Filter controls."""

from __future__ import annotations

from dash import html

from maths_self_study.dashboards.components import filter_bar, slider


def build_filters() -> html.Div:
    return filter_bar(
        slider("pc-biplot-x", "Biplot PC x", 1, 5, 1, 1),
        slider("pc-biplot-y", "Biplot PC y", 1, 5, 2, 1),
    )
