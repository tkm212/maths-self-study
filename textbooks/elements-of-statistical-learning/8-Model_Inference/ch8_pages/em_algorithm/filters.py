"""Filter controls."""

from __future__ import annotations

from dash import html

from maths_self_study.dashboards.components import filter_bar, slider


def build_filters() -> html.Div:
    return filter_bar(
        slider("em-n-samples", "Samples", 200, 1000, 500, 50),
        slider("em-k", "Components K", 2, 4, 2, 1),
        slider("em-restarts", "Random restarts", 3, 10, 5, 1),
    )
