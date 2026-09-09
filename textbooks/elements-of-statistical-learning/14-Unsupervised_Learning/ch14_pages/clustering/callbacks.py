"""Dash callbacks for clustering page."""

from __future__ import annotations

from dash import Input

from ch14_pages.clustering.content import render_body
from maths_self_study.dashboards.callbacks import define_page_callbacks

INPUTS = [
    Input("cl-k-max", "value"),
    Input("cl-centroid-k", "value"),
    Input("cl-linkage", "value"),
    Input("cl-linkage-k", "value"),
]

register_callbacks = define_page_callbacks(render_body=render_body, inputs=INPUTS, page="clustering")
