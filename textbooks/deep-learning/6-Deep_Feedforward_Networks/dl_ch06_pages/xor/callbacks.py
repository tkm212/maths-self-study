"""Dash callbacks for the XOR page."""

from __future__ import annotations

from dash import Input

from dl_ch06_pages.xor.content import render_body
from maths_self_study.dashboards.callbacks import define_page_callbacks

INPUTS = [
    Input("xor-hidden", "value"),
    Input("xor-lr", "value"),
    Input("xor-epochs", "value"),
    Input("xor-activation", "value"),
]

register_callbacks = define_page_callbacks(
    render_body=render_body,
    inputs=INPUTS,
    page="xor",
)
