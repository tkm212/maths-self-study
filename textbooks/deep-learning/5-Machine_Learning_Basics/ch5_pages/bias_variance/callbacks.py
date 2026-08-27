"""Dash callbacks for the bias-variance page."""

from __future__ import annotations

from dash import Input

from ch5_pages.bias_variance.content import render_body
from maths_self_study.dashboards.callbacks import define_page_callbacks

INPUTS = [
    Input("bv-degree", "value"),
]

register_callbacks = define_page_callbacks(
    render_body=render_body,
    inputs=INPUTS,
    page="bias_variance",
)
