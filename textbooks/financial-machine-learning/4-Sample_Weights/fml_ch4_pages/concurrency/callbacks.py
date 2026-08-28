"""Dash callbacks for the concurrency page."""

from __future__ import annotations

from dash import Input

from fml_ch4_pages.concurrency.content import render_body
from maths_self_study.dashboards.callbacks import define_page_callbacks

INPUTS = [
    Input("weight-cusum", "value"),
    Input("weight-pt", "value"),
    Input("weight-sl", "value"),
    Input("weight-num-bars", "value"),
    Input("weight-max-points", "value"),
]

register_callbacks = define_page_callbacks(
    render_body=render_body,
    inputs=INPUTS,
    page="concurrency",
)
