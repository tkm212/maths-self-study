"""Dash callbacks for the receptive field page."""

from __future__ import annotations

from dash import Input

from dl_ch09_pages.receptive_field.content import render_body
from maths_self_study.dashboards.callbacks import define_page_callbacks

INPUTS = [
    Input("rf-blocks", "value"),
    Input("rf-kernel", "value"),
]

register_callbacks = define_page_callbacks(
    render_body=render_body,
    inputs=INPUTS,
    page="receptive_field",
)
