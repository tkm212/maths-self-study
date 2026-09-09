"""Dash callbacks for neural networks page."""

from __future__ import annotations

from dash import Input

from ch11_pages.neural_networks.content import render_body
from maths_self_study.dashboards.callbacks import define_page_callbacks

INPUTS = [
    Input("nn-max-epochs", "value"),
    Input("nn-hidden-units", "value"),
]

register_callbacks = define_page_callbacks(render_body=render_body, inputs=INPUTS, page="neural_networks")
