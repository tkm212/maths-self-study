"""Filter controls for the sample weights page."""

from __future__ import annotations

from dash import html

from maths_self_study.dashboards.components import filter_bar, num_input, slider


def build_filters() -> html.Div:
    return filter_bar(
        slider("sw-cusum", "CUSUM threshold", 0.00005, 0.001, 0.0002, 0.00005),
        slider("sw-pt", "Profit take", 0.0005, 0.005, 0.001, 0.0005),
        slider("sw-sl", "Stop loss", 0.0005, 0.005, 0.001, 0.0005),
        num_input("sw-num-bars", "Max bars", 30, step=5),
        slider("sw-decay-hours", "Decay span (hours)", 0.25, 6.0, 1.0, 0.25),
    )
