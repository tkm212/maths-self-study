"""Dash callbacks for random forests page."""

from __future__ import annotations

from dash import Input

from ch15_pages.random_forests.content import render_body
from maths_self_study.dashboards.callbacks import define_page_callbacks

INPUTS = [Input("rf-n-estimators", "value")]

register_callbacks = define_page_callbacks(render_body=render_body, inputs=INPUTS, page="random_forests")
