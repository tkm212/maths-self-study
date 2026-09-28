"""Dash callbacks for the convolution page."""

from __future__ import annotations

from dash import Input

from dl_ch09_pages.convolution.content import render_body
from maths_self_study.dashboards.callbacks import define_page_callbacks

INPUTS = [
    Input("conv-filter", "value"),
    Input("conv-stride", "value"),
    Input("conv-padding", "value"),
]

register_callbacks = define_page_callbacks(
    render_body=render_body,
    inputs=INPUTS,
    page="convolution",
)
