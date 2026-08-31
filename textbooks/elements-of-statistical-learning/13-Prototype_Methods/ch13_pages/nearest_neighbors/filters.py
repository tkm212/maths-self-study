"""Filter controls."""

from __future__ import annotations

from dash import html

from maths_self_study.dashboards.components import filter_bar, slider


def build_filters() -> html.Div:
    return filter_bar(
        slider("knn-metric-k", "Metric comparison k", 1, 20, 5, 1),
        slider("knn-max-k", "Train/test max k", 20, 50, 50, 5),
    )
