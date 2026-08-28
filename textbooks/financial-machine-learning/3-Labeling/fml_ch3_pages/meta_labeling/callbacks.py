"""Dash callbacks for the meta-labeling page."""

from __future__ import annotations

from dash import Input

from fml_ch3_pages.meta_labeling.content import render_body
from maths_self_study.dashboards.callbacks import define_page_callbacks

INPUTS = [Input("meta-refresh", "value")]

register_callbacks = define_page_callbacks(
    render_body=lambda _refresh: render_body(),
    inputs=INPUTS,
    page="meta_labeling",
)
