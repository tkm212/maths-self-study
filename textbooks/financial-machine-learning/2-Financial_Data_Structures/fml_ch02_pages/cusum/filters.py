"""Filter controls for the CUSUM page."""

from __future__ import annotations

from dash import html

from maths_self_study.dashboards.components import filter_bar, slider


def build_filters() -> html.Div:
    return filter_bar(
        slider("cusum-threshold", "CUSUM threshold", 0.00005, 0.001, 0.0002, 0.00005),
    )
