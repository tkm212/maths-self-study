"""Dash callbacks for cross-validation page."""

from __future__ import annotations

from dash import Input

from ch7_pages.cross_validation.content import render_body
from maths_self_study.dashboards.callbacks import define_page_callbacks

INPUTS = [Input("cv-feat", "value"), Input("cv-max-degree", "value")]

register_callbacks = define_page_callbacks(render_body=render_body, inputs=INPUTS, page="cross_validation")
