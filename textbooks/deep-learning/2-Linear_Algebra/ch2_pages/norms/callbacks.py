"""Dash callbacks for the norms page."""

from __future__ import annotations

from dash import Input

from ch2_pages.norms.content import render_body
from maths_self_study.dashboards.callbacks import define_page_callbacks

INPUTS = [
    Input("norm-x1", "value"),
    Input("norm-x2", "value"),
    Input("norm-inf", "value"),
]

register_callbacks = define_page_callbacks(
    render_body=render_body,
    inputs=INPUTS,
    page="norms",
)
