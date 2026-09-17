"""Dash callbacks for the tangent distance page."""

from __future__ import annotations

from dash import Input

from dl_ch07_pages.tangent_distance.content import render_body
from maths_self_study.dashboards.callbacks import define_page_callbacks

INPUTS = [Input("td-shift", "value")]

register_callbacks = define_page_callbacks(
    render_body=render_body,
    inputs=INPUTS,
    page="tangent_distance",
)
