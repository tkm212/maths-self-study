"""Dash callbacks for the PCA weights page."""

from __future__ import annotations

from dash import Input

from fml_ch2_pages.pca_weights.content import render_body
from maths_self_study.dashboards.callbacks import define_page_callbacks

INPUTS = [Input("pca-component", "value")]

register_callbacks = define_page_callbacks(
    render_body=render_body,
    inputs=INPUTS,
    page="pca_weights",
)
