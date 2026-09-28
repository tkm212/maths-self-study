"""Dash callbacks for the mini-batch page."""

from __future__ import annotations

from dash import Input

from dl_ch08_pages.minibatch.content import render_body
from maths_self_study.dashboards.callbacks import define_page_callbacks

INPUTS = [Input("mb-size", "value")]

register_callbacks = define_page_callbacks(
    render_body=render_body,
    inputs=INPUTS,
    page="minibatch",
)
