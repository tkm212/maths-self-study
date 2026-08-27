"""Dash callbacks for the capacity page."""

from __future__ import annotations

from dash import Input

from ch5_pages.capacity.content import render_body
from maths_self_study.dashboards.callbacks import define_page_callbacks

INPUTS = [
    Input("cap-degree", "value"),
    Input("cap-noise", "value"),
]

register_callbacks = define_page_callbacks(
    render_body=render_body,
    inputs=INPUTS,
    page="capacity",
)
