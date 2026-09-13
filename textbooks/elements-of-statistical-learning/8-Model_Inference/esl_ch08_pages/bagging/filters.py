"""Filter controls."""

from __future__ import annotations

from dash import html

from maths_self_study.dashboards.components import dropdown, filter_bar, slider

FEAT_OPTIONS = [
    {"label": "Budget", "value": "budget"},
    {"label": "Popularity", "value": "popularity"},
    {"label": "Votes", "value": "vote_count"},
    {"label": "Runtime", "value": "runtime"},
    {"label": "Release year", "value": "release_year"},
]


def build_filters() -> html.Div:
    return filter_bar(
        dropdown("bag-feat", "Feature", FEAT_OPTIONS, "budget"),
        slider("bag-degree", "Polynomial degree", 1, 5, 3, 1),
        slider("bag-tree-depth", "Tree max depth", 2, 10, 5, 1),
        slider("bag-max-bags", "Max ensemble size B", 10, 50, 50, 5),
    )
