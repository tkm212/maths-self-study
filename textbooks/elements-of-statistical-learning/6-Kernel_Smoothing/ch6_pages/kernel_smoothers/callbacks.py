"""Dash callbacks for kernel smoothers page."""

from __future__ import annotations

from dash import Input

from ch6_pages.kernel_smoothers.content import render_body
from maths_self_study.dashboards.callbacks import define_page_callbacks

INPUTS = [Input("ks-feat", "value"), Input("ks-bw", "value")]

register_callbacks = define_page_callbacks(render_body=render_body, inputs=INPUTS, page="kernel_smoothers")
