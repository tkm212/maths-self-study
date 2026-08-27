"""Dash callbacks for the gradient descent page."""

from __future__ import annotations

from dash import Input

from ch4_pages.gradient_descent.content import render_body
from maths_self_study.dashboards.callbacks import define_page_callbacks

INPUTS = [
    Input("gd-eta", "value"),
    Input("gd-x0", "value"),
    Input("gd-x1", "value"),
]

register_callbacks = define_page_callbacks(
    render_body=render_body,
    inputs=INPUTS,
    page="gradient_descent",
)
