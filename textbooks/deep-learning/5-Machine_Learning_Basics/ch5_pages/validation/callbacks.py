"""Dash callbacks for the validation page."""

from __future__ import annotations

from dash import Input

from ch5_pages.validation.content import render_body
from maths_self_study.dashboards.callbacks import define_page_callbacks

INPUTS = [
    Input("val-l2", "value"),
]

register_callbacks = define_page_callbacks(
    render_body=render_body,
    inputs=INPUTS,
    page="validation",
)
