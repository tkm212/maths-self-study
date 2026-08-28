"""Dash callbacks for separating hyperplanes page."""

from __future__ import annotations

from dash import Input

from ch4_pages.separating_hyperplanes.content import render_body
from maths_self_study.dashboards.callbacks import define_page_callbacks

INPUTS = [
    Input("sh-n", "value"),
    Input("sh-margin", "value"),
    Input("sh-seed", "value"),
]

register_callbacks = define_page_callbacks(
    render_body=render_body,
    inputs=INPUTS,
    page="separating_hyperplanes",
)
