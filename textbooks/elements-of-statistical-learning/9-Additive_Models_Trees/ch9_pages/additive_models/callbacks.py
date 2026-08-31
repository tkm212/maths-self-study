"""Dash callbacks for additive models page."""

from __future__ import annotations

from dash import Input

from ch9_pages.additive_models.content import render_body
from maths_self_study.dashboards.callbacks import define_page_callbacks

INPUTS = [Input("gam-n-knots", "value")]

register_callbacks = define_page_callbacks(render_body=render_body, inputs=INPUTS, page="additive_models")
