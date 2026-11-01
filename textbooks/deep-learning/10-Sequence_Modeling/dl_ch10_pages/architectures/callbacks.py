"""Dash callbacks for the architectures page."""

from __future__ import annotations

from dash import Input

from dl_ch10_pages.architectures.content import render_body
from maths_self_study.dashboards.callbacks import define_page_callbacks

INPUTS = [
    Input("arch-length", "value"),
    Input("arch-context", "value"),
    Input("arch-stack", "value"),
]

register_callbacks = define_page_callbacks(
    render_body=render_body,
    inputs=INPUTS,
    page="architectures",
)
