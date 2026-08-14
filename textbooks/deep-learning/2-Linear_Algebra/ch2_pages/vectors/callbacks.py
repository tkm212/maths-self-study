"""Dash callbacks for the vectors page."""

from __future__ import annotations

from dash import Dash, Input

from ch2_pages.vectors.content import render_body
from maths_self_study.dashboards.callbacks import register_body_callback

INPUTS = [
    Input("grid-a11", "value"),
    Input("grid-a12", "value"),
    Input("grid-a21", "value"),
    Input("grid-a22", "value"),
    Input("vm-rot", "value"),
    Input("vm-shear", "value"),
    Input("vm-range", "value"),
]


def register_callbacks(app: Dash, body_id: str) -> None:
    register_body_callback(app, body_id, INPUTS, render_body, page="vectors")
