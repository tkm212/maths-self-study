"""Filter controls for the k-NN page."""

from __future__ import annotations

from dash import html

from maths_self_study.dashboards.components import filter_bar, num_input, slider


def build_filters() -> html.Div:
    return filter_bar(
        num_input("knn-max-rows", "Max rows", 40_000, step=5_000, min_=5_000),
        slider("knn-k", "k for single-feature plot", 3, 25, 15, 1),
    )
