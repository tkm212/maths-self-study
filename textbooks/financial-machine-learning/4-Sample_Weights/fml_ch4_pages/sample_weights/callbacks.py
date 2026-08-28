"""Dash callbacks for the sample weights page."""

from __future__ import annotations

from dash import Input

from fml_ch4_pages.sample_weights.content import render_body
from maths_self_study.dashboards.callbacks import define_page_callbacks

INPUTS = [
    Input("sw-cusum", "value"),
    Input("sw-pt", "value"),
    Input("sw-sl", "value"),
    Input("sw-num-bars", "value"),
    Input("sw-decay-hours", "value"),
]

register_callbacks = define_page_callbacks(
    render_body=render_body,
    inputs=INPUTS,
    page="sample_weights",
)
