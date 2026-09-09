"""Filter controls."""

from __future__ import annotations

from dash import html

from maths_self_study.dashboards.components import filter_bar, slider


def build_filters() -> html.Div:
    return filter_bar(
        slider("nn-max-epochs", "Max epochs", 50, 300, 150, 25),
        slider("nn-hidden-units", "Hidden units M", 10, 100, 50, 10),
    )
