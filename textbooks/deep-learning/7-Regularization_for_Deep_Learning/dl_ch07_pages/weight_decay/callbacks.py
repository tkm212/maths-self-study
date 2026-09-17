"""Dash callbacks for the weight decay page."""

from __future__ import annotations

from dash import Input

from dl_ch07_pages.weight_decay.content import render_body
from maths_self_study.dashboards.callbacks import define_page_callbacks

INPUTS = [Input("wd-l2", "value")]

register_callbacks = define_page_callbacks(
    render_body=render_body,
    inputs=INPUTS,
    page="weight_decay",
)
