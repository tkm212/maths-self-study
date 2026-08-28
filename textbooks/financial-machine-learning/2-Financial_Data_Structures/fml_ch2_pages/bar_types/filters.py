"""Filter controls for the bar types page."""

from __future__ import annotations

from dash import html

from maths_self_study.dashboards.components import filter_bar, num_input, slider


def build_filters() -> html.Div:
    return filter_bar(
        num_input("bars-tick-threshold", "Ticks per bar", 100, step=10),
        num_input("bars-target", "Target bar count (vol/dollar)", 300, step=50),
        slider("bars-save", "Save to outputs", 0, 1, 1, 1),
    )
