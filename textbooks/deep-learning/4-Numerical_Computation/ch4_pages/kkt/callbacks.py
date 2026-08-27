"""Dash callbacks for the KKT conditions page."""

from __future__ import annotations

from dash import Input

from ch4_pages.kkt.content import render_body
from maths_self_study.dashboards.callbacks import define_page_callbacks

INPUTS = [
    Input("kkt-bound", "value"),
]

register_callbacks = define_page_callbacks(
    render_body=render_body,
    inputs=INPUTS,
    page="kkt",
)
