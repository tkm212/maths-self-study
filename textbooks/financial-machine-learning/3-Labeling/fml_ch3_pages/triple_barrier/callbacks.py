"""Dash callbacks for the triple-barrier page."""

from __future__ import annotations

from dash import Input

from fml_ch3_pages.triple_barrier.content import render_body
from maths_self_study.dashboards.callbacks import define_page_callbacks

INPUTS = [
    Input("label-cusum", "value"),
    Input("label-pt", "value"),
    Input("label-sl", "value"),
    Input("label-num-bars", "value"),
    Input("label-sample", "value"),
]

register_callbacks = define_page_callbacks(
    render_body=render_body,
    inputs=INPUTS,
    page="triple_barrier",
)
