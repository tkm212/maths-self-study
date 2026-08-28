"""Dash callbacks for the CUSUM page."""

from __future__ import annotations

from dash import Input

from fml_ch2_pages.cusum.content import render_body
from maths_self_study.dashboards.callbacks import define_page_callbacks

INPUTS = [Input("cusum-threshold", "value")]

register_callbacks = define_page_callbacks(
    render_body=render_body,
    inputs=INPUTS,
    page="cusum",
)
