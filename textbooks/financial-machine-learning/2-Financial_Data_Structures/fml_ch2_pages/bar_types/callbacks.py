"""Dash callbacks for the bar types page."""

from __future__ import annotations

from dash import Input

from fml_ch2_pages.bar_types.content import render_body
from maths_self_study.dashboards.callbacks import define_page_callbacks

INPUTS = [
    Input("bars-tick-threshold", "value"),
    Input("bars-target", "value"),
    Input("bars-save", "value"),
]

register_callbacks = define_page_callbacks(
    render_body=render_body,
    inputs=INPUTS,
    page="bar_types",
)
