"""Dash callbacks for splines page."""

from __future__ import annotations

from dash import Input

from ch5_pages.splines.content import render_body
from maths_self_study.dashboards.callbacks import define_page_callbacks

INPUTS = [Input("spl-feat", "value"), Input("spl-n-knots", "value")]

register_callbacks = define_page_callbacks(render_body=render_body, inputs=INPUTS, page="splines")
