"""Dash callbacks for the backprop page."""

from __future__ import annotations

from dash import Input

from dl_ch06_pages.backprop.content import render_body
from maths_self_study.dashboards.callbacks import define_page_callbacks

INPUTS = [
    Input("bp-hidden", "value"),
    Input("bp-epsilon", "value"),
    Input("bp-sample", "value"),
    Input("bp-activation", "value"),
]

register_callbacks = define_page_callbacks(
    render_body=render_body,
    inputs=INPUTS,
    page="backprop",
)
