"""Filter controls."""

from __future__ import annotations

from dash import html

from maths_self_study.dashboards.components import filter_bar


def build_filters() -> html.Div:
    return filter_bar()
