"""Dash callbacks for the output units page."""

from __future__ import annotations

from dash import Input

from dl_ch06_pages.output_units.content import render_body
from maths_self_study.dashboards.callbacks import define_page_callbacks

INPUTS = [
    Input("out-x", "value"),
    Input("out-z0", "value"),
    Input("out-z1", "value"),
    Input("out-z2", "value"),
]

register_callbacks = define_page_callbacks(
    render_body=render_body,
    inputs=INPUTS,
    page="output_units",
)
