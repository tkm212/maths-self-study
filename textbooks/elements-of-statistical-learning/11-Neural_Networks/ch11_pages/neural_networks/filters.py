"""Filter controls."""

from __future__ import annotations

from dash import html

from maths_self_study.dashboards.components import filter_bar, slider


def build_filters() -> html.Div:
    return filter_bar(
        slider("nn-max-epochs", "Max epochs", 25, 150, 80, 25),
        slider("nn-hidden-units", "Hidden units M", 5, 50, 20, 5),
    )
