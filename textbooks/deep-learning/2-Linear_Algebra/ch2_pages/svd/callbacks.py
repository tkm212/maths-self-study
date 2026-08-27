"""Dash callbacks for the SVD page."""

from __future__ import annotations

from dash import Input

from ch2_pages.svd.content import render_body
from maths_self_study.dashboards.callbacks import define_page_callbacks
from maths_self_study.dashboards.components import matrix_callback_inputs

INPUTS = [
    *matrix_callback_inputs("svd-matrix"),
    Input("svd-b0", "value"),
    Input("svd-b1", "value"),
    Input("svd-b2", "value"),
]

register_callbacks = define_page_callbacks(
    render_body=render_body,
    inputs=INPUTS,
    page="svd",
)
