"""Dash callbacks for the k-NN page."""

from __future__ import annotations

from dash import Input

from ch2_pages.k_nearest_neighbors.content import render_body
from maths_self_study.dashboards.callbacks import define_page_callbacks

INPUTS = [
    Input("knn-max-rows", "value"),
    Input("knn-k", "value"),
]

register_callbacks = define_page_callbacks(
    render_body=render_body,
    inputs=INPUTS,
    page="k_nearest_neighbors",
)
