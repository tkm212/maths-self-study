"""Filter controls for the norms page."""

from __future__ import annotations

from dash import html

from maths_self_study.dashboards.components import checklist, filter_bar, num_input


def build_filters() -> html.Div:
    return filter_bar(
        num_input("norm-x1", "x₁", 3.0, step=0.5),
        num_input("norm-x2", "x₂", -4.0, step=0.5),
        checklist("norm-inf", "Include L∞", [{"label": " L∞", "value": "inf"}], ["inf"]),
    )
