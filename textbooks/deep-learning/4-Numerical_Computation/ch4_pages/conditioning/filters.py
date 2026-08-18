"""Filter controls for the conditioning page."""

from __future__ import annotations

from dash import html

from maths_self_study.dashboards.components import filter_bar, num_input, slider


def build_filters() -> html.Div:
    return filter_bar(
        slider("cond-kappa", "Condition number ratio", 1.0, 1000.0, 10.0, step=1.0),
        num_input("cond-delta", "Perturbation δ", 1e-6, step=1e-7),
    )
