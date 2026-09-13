"""Dash callbacks for the universal approximation page."""

from __future__ import annotations

from dash import Input

from dl_ch06_pages.universal_approx.content import render_body
from maths_self_study.dashboards.callbacks import define_page_callbacks

INPUTS = [
    Input("approx-hidden", "value"),
    Input("approx-noise", "value"),
    Input("approx-activation", "value"),
]

register_callbacks = define_page_callbacks(
    render_body=render_body,
    inputs=INPUTS,
    page="universal_approx",
)
