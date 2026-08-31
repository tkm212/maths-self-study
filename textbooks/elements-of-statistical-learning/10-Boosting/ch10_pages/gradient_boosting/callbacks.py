"""Dash callbacks for gradient boosting page."""

from __future__ import annotations

from dash import Input

from ch10_pages.gradient_boosting.content import render_body
from maths_self_study.dashboards.callbacks import define_page_callbacks

INPUTS = [
    Input("gbm-n-estimators", "value"),
    Input("gbm-learning-rate", "value"),
    Input("gbm-max-depth", "value"),
]

register_callbacks = define_page_callbacks(render_body=render_body, inputs=INPUTS, page="gradient_boosting")
