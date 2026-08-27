"""Dash callbacks for the SGD page."""

from __future__ import annotations

from dash import Input

from ch5_pages.sgd.content import render_body
from maths_self_study.dashboards.callbacks import define_page_callbacks

INPUTS = [
    Input("sgd-eta", "value"),
    Input("sgd-batch", "value"),
]

register_callbacks = define_page_callbacks(
    render_body=render_body,
    inputs=INPUTS,
    page="sgd",
)
