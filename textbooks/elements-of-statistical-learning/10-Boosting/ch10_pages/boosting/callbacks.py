"""Dash callbacks for AdaBoost page."""

from __future__ import annotations

from dash import Input

from ch10_pages.boosting.content import render_body
from maths_self_study.dashboards.callbacks import define_page_callbacks

INPUTS = [Input("ada-n-estimators", "value")]

register_callbacks = define_page_callbacks(render_body=render_body, inputs=INPUTS, page="boosting")
