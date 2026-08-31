"""Dash callbacks for EM algorithm page."""

from __future__ import annotations

from dash import Input

from ch8_pages.em_algorithm.content import render_body
from maths_self_study.dashboards.callbacks import define_page_callbacks

INPUTS = [
    Input("em-n-samples", "value"),
    Input("em-k", "value"),
    Input("em-restarts", "value"),
]

register_callbacks = define_page_callbacks(render_body=render_body, inputs=INPUTS, page="em_algorithm")
