"""Dash callbacks for nearest neighbors page."""

from __future__ import annotations

from dash import Input

from ch13_pages.nearest_neighbors.content import render_body
from maths_self_study.dashboards.callbacks import define_page_callbacks

INPUTS = [Input("knn-metric-k", "value"), Input("knn-max-k", "value")]

register_callbacks = define_page_callbacks(render_body=render_body, inputs=INPUTS, page="nearest_neighbors")
