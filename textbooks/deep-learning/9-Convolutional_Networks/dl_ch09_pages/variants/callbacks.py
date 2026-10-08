"""Dash callbacks for the variants page."""

from __future__ import annotations

from dash import Input

from dl_ch09_pages.variants.content import render_body
from maths_self_study.dashboards.callbacks import define_page_callbacks

INPUTS = [Input("var-dilation", "value")]

register_callbacks = define_page_callbacks(
    render_body=render_body,
    inputs=INPUTS,
    page="variants",
)
