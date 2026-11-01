"""Dash callbacks for the unfolding page."""

from __future__ import annotations

from dash import Input

from dl_ch10_pages.unfolding.content import render_body
from maths_self_study.dashboards.callbacks import define_page_callbacks

INPUTS = [
    Input("unfold-theta", "value"),
    Input("unfold-gain", "value"),
    Input("unfold-pulse", "value"),
]

register_callbacks = define_page_callbacks(
    render_body=render_body,
    inputs=INPUTS,
    page="unfolding",
)
