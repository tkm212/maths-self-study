"""Filter controls for the concurrency page."""

from __future__ import annotations

from dash import html

from maths_self_study.dashboards.components import filter_bar, num_input, slider


def build_filters() -> html.Div:
    return filter_bar(
        slider("weight-cusum", "CUSUM threshold", 0.00005, 0.001, 0.0002, 0.00005),
        slider("weight-pt", "Profit take", 0.0005, 0.005, 0.001, 0.0005),
        slider("weight-sl", "Stop loss", 0.0005, 0.005, 0.001, 0.0005),
        num_input("weight-num-bars", "Max bars", 30, step=5),
        num_input("weight-max-points", "Chart bars", 5000, step=500),
    )
