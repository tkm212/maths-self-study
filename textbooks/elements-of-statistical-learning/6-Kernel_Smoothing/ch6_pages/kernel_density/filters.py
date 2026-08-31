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
        dropdown("kd-feat", "Feature", FEAT_OPTIONS, "budget"),
        slider("kd-bw", "Naive Bayes bandwidth", 0.05, 1.0, 0.3, 0.05),
    )
