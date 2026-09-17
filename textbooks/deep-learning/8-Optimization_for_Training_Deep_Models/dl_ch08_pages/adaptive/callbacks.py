"""Dash callbacks for the adaptive optimizers page."""

from __future__ import annotations

from dash import Input

from dl_ch08_pages.adaptive.content import render_body
from maths_self_study.dashboards.callbacks import define_page_callbacks

INPUTS = [Input("adapt-lr", "value")]

register_callbacks = define_page_callbacks(
    render_body=render_body,
    inputs=INPUTS,
    page="adaptive",
)
