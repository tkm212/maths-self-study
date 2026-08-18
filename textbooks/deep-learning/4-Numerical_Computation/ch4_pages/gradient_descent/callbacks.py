"""Dash callbacks for the gradient descent page."""

from __future__ import annotations

from dash import Dash, Input

from ch4_pages.gradient_descent.content import render_body
from maths_self_study.dashboards.callbacks import register_body_callback

INPUTS = [
    Input("gd-eta", "value"),
    Input("gd-x0", "value"),
    Input("gd-x1", "value"),
]


def register_callbacks(app: Dash, body_id: str) -> None:
    register_body_callback(app, body_id, INPUTS, render_body, page="gradient_descent")
