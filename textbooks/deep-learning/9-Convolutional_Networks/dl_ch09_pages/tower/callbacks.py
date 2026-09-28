"""Dash callbacks for the tower page."""

from __future__ import annotations

from dash import Input

from dl_ch09_pages.tower.content import render_body
from maths_self_study.dashboards.callbacks import define_page_callbacks

INPUTS = [
    Input("tower-blocks", "value"),
    Input("tower-kernel", "value"),
]

register_callbacks = define_page_callbacks(
    render_body=render_body,
    inputs=INPUTS,
    page="tower",
)
