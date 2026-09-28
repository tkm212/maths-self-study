"""Dash callbacks for the translation page."""

from __future__ import annotations

from dash import Input

from dl_ch09_pages.translation.content import render_body
from maths_self_study.dashboards.callbacks import define_page_callbacks

INPUTS = [
    Input("trans-shift", "value"),
    Input("trans-padding", "value"),
]

register_callbacks = define_page_callbacks(
    render_body=render_body,
    inputs=INPUTS,
    page="translation",
)
