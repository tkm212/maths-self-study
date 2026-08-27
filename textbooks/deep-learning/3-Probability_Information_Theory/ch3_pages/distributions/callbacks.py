"""Dash callbacks for the distributions page."""

from __future__ import annotations

from dash import Input

from ch3_pages.distributions.content import render_body
from maths_self_study.dashboards.callbacks import define_page_callbacks
from maths_self_study.dashboards.components import matrix_callback_inputs

INPUTS = [
    Input("dist-c0", "value"),
    Input("dist-c1", "value"),
    Input("dist-c2", "value"),
    Input("dist-c3", "value"),
    *matrix_callback_inputs("dist-cov-matrix"),
]

register_callbacks = define_page_callbacks(
    render_body=render_body,
    inputs=INPUTS,
    page="distributions",
)
