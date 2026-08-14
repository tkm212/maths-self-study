"""Dash callbacks for the PCA page."""

from __future__ import annotations

from dash import Dash, Input

from ch2_pages.pca.content import render_body
from maths_self_study.dashboards.callbacks import register_body_callback

INPUTS = [
    Input("pca-seed", "value"),
    Input("pca-n", "value"),
    Input("pca-sx", "value"),
    Input("pca-sy", "value"),
]


def register_callbacks(app: Dash, body_id: str) -> None:
    register_body_callback(app, body_id, INPUTS, render_body, page="pca")
