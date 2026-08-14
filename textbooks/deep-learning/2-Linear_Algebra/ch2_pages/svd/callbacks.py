"""Dash callbacks for the SVD page."""

from __future__ import annotations

from dash import Dash, Input

from ch2_pages.svd.content import render_body
from maths_self_study.dashboards.callbacks import register_body_callback

INPUTS = [
    Input("svd-a11", "value"),
    Input("svd-a12", "value"),
    Input("svd-a21", "value"),
    Input("svd-a22", "value"),
    Input("svd-b0", "value"),
    Input("svd-b1", "value"),
    Input("svd-b2", "value"),
]


def register_callbacks(app: Dash, body_id: str) -> None:
    register_body_callback(app, body_id, INPUTS, render_body, page="svd")
