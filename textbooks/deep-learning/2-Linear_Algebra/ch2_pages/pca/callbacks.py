"""Dash callbacks for the PCA page."""

from __future__ import annotations

from dash import Input

from ch2_pages.pca.content import render_body
from maths_self_study.dashboards.callbacks import define_page_callbacks

INPUTS = [
    Input("pca-seed", "value"),
    Input("pca-n", "value"),
    Input("pca-sx", "value"),
    Input("pca-sy", "value"),
]

register_callbacks = define_page_callbacks(
    render_body=render_body,
    inputs=INPUTS,
    page="pca",
)
