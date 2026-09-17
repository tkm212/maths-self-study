"""Dash callbacks for the input noise page."""

from __future__ import annotations

from dash import Input

from dl_ch07_pages.input_noise.content import render_body
from maths_self_study.dashboards.callbacks import define_page_callbacks

INPUTS = [Input("in-noise", "value")]

register_callbacks = define_page_callbacks(
    render_body=render_body,
    inputs=INPUTS,
    page="input_noise",
)
