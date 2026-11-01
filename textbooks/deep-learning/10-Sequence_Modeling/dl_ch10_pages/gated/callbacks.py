"""Dash callbacks for the gated RNN page."""

from __future__ import annotations

from dash import Input

from dl_ch10_pages.gated.content import render_body
from maths_self_study.dashboards.callbacks import define_page_callbacks

INPUTS = [
    Input("gated-forget", "value"),
    Input("gated-cue", "value"),
    Input("gated-query", "value"),
]

register_callbacks = define_page_callbacks(
    render_body=render_body,
    inputs=INPUTS,
    page="gated",
)
