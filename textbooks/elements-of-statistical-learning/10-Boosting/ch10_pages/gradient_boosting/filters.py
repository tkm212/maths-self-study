"""Filter controls."""

from __future__ import annotations

from dash import html

from maths_self_study.dashboards.components import filter_bar, slider


def build_filters() -> html.Div:
    return filter_bar(
        slider("gbm-n-estimators", "Trees M", 50, 300, 200, 25),
        slider("gbm-learning-rate", "Learning rate ν", 0.01, 0.5, 0.1, 0.01),
        slider("gbm-max-depth", "Tree depth", 1, 6, 3, 1),
    )
