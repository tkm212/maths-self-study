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
        dropdown("ks-feat", "Feature", FEAT_OPTIONS, "budget"),
        slider("ks-bw", "Bandwidth (LL vs NW)", 0.1, 1.5, 0.5, 0.05),
    )
