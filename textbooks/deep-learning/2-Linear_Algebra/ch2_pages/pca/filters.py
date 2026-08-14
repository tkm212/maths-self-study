"""Filter controls for the PCA page."""

from __future__ import annotations

from dash import html

from maths_self_study.dashboards.components import filter_bar, num_input, slider


def build_filters() -> html.Div:
    return filter_bar(
        num_input("pca-seed", "RNG seed", 42, step=1),
        num_input("pca-n", "Samples", 300, step=50),
        slider("pca-sx", "Stretch x", 0.5, 5.0, 3.0, 0.1),
        slider("pca-sy", "Stretch y", 0.1, 2.0, 0.5, 0.1),
    )
