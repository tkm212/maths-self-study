"""Dash callbacks for the conditioning page."""

from __future__ import annotations

from dash import Input

from ch4_pages.conditioning.content import render_body
from maths_self_study.dashboards.callbacks import define_page_callbacks

INPUTS = [
    Input("cond-kappa", "value"),
    Input("cond-delta", "value"),
]

register_callbacks = define_page_callbacks(
    render_body=render_body,
    inputs=INPUTS,
    page="conditioning",
)
