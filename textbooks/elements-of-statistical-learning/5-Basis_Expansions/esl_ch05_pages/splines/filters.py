"""Filter controls."""

from __future__ import annotations

from dash import html

from maths_self_study.dashboards.components import dropdown, filter_bar, num_input

FEAT_OPTIONS = [
    {"label": "Budget", "value": "budget"},
    {"label": "Popularity", "value": "popularity"},
    {"label": "Votes", "value": "vote_count"},
    {"label": "Runtime", "value": "runtime"},
    {"label": "Release year", "value": "release_year"},
]


def build_filters() -> html.Div:
    return filter_bar(
        dropdown("spl-feat", "Feature", FEAT_OPTIONS, "budget"),
        num_input("spl-n-knots", "Knots (piecewise)", 4, step=1, min_=2),
    )
