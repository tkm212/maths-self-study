"""Dash callbacks for the Newton page."""

from __future__ import annotations

from dash import Input

from ch4_pages.newton.content import render_body
from maths_self_study.dashboards.callbacks import define_page_callbacks

INPUTS = [
    Input("newton-eta", "value"),
    Input("newton-x0", "value"),
    Input("newton-x1", "value"),
]

register_callbacks = define_page_callbacks(
    render_body=render_body,
    inputs=INPUTS,
    page="newton",
)
