"""Dash callbacks for projection pursuit page."""

from __future__ import annotations

from dash import Input

from ch11_pages.projection_pursuit.content import render_body
from maths_self_study.dashboards.callbacks import define_page_callbacks

INPUTS = [Input("ppr-ridge-m", "value")]

register_callbacks = define_page_callbacks(render_body=render_body, inputs=INPUTS, page="projection_pursuit")
