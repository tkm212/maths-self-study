"""Dash callbacks for the momentum page."""

from __future__ import annotations

from dash import Input

from dl_ch08_pages.momentum.content import render_body
from maths_self_study.dashboards.callbacks import define_page_callbacks

INPUTS = [
    Input("mom-lr", "value"),
    Input("mom-rho", "value"),
]

register_callbacks = define_page_callbacks(
    render_body=render_body,
    inputs=INPUTS,
    page="momentum",
)
