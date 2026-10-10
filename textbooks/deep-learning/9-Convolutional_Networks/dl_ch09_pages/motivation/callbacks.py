"""Dash callbacks for the motivation page."""

from __future__ import annotations

from dash import Input

from dl_ch09_pages.motivation.content import render_body
from maths_self_study.dashboards.callbacks import define_page_callbacks

INPUTS = [
    Input("mot-filter", "value"),
    Input("mot-shift", "value"),
    Input("mot-blocks", "value"),
]

register_callbacks = define_page_callbacks(
    render_body=render_body,
    inputs=INPUTS,
    page="motivation",
)
