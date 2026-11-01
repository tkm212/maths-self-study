"""Dash callbacks for the long-term dependencies page."""

from __future__ import annotations

from dash import Input

from dl_ch10_pages.dependencies.content import render_body
from maths_self_study.dashboards.callbacks import define_page_callbacks

INPUTS = [
    Input("dep-magnitude", "value"),
    Input("dep-alpha", "value"),
    Input("dep-radius", "value"),
    Input("dep-clip", "value"),
]

register_callbacks = define_page_callbacks(
    render_body=render_body,
    inputs=INPUTS,
    page="dependencies",
)
