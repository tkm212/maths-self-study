"""Dash callbacks for the eigendecomposition page."""

from __future__ import annotations

from dash import Dash, Input

from ch2_pages.eigendecomposition.content import render_body
from maths_self_study.dashboards.callbacks import register_body_callback

INPUTS = [
    Input("cov-a11", "value"),
    Input("cov-a12", "value"),
    Input("cov-a21", "value"),
    Input("cov-a22", "value"),
]


def register_callbacks(app: Dash, body_id: str) -> None:
    register_body_callback(app, body_id, INPUTS, render_body, page="eigendecomposition")
