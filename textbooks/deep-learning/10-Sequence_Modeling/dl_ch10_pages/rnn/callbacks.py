"""Dash callbacks for the RNN page."""

from __future__ import annotations

from dash import Input

from dl_ch10_pages.rnn.content import render_body
from maths_self_study.dashboards.callbacks import define_page_callbacks

INPUTS = [
    Input("rnn-weight", "value"),
    Input("rnn-cue", "value"),
]

register_callbacks = define_page_callbacks(
    render_body=render_body,
    inputs=INPUTS,
    page="rnn",
)
