"""Dash callbacks for the vectors page."""

from __future__ import annotations

from dash import Input

from ch2_pages.vectors.content import render_body
from maths_self_study.dashboards.callbacks import define_page_callbacks
from maths_self_study.dashboards.components import matrix_callback_inputs

INPUTS = [
    *matrix_callback_inputs("grid-matrix"),
    Input("vm-rot", "value"),
    Input("vm-shear", "value"),
]

register_callbacks = define_page_callbacks(
    render_body=render_body,
    inputs=INPUTS,
    page="vectors",
)
