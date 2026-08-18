"""Dash callbacks for the Newton page."""

from __future__ import annotations

from dash import Dash, Input

from ch4_pages.newton.content import render_body
from maths_self_study.dashboards.callbacks import register_body_callback

INPUTS = [
    Input("newton-eta", "value"),
    Input("newton-x0", "value"),
    Input("newton-x1", "value"),
]


def register_callbacks(app: Dash, body_id: str) -> None:
    register_body_callback(app, body_id, INPUTS, render_body, page="newton")
