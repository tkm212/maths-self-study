"""Filter controls for the PCA weights page."""

from __future__ import annotations

from dash import html

from maths_self_study.dashboards.components import filter_bar, slider


def build_filters() -> html.Div:
    return filter_bar(
        slider("pca-component", "Principal component", 1, 4, 1, 1),
    )
