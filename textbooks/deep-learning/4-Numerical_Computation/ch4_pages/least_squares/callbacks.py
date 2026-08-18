"""Dash callbacks for the least squares page."""

from __future__ import annotations

from dash import Dash, Input

from ch4_pages.least_squares.content import render_body
from maths_self_study.dashboards.callbacks import register_body_callback

INPUTS = [
    Input("ls-y0", "value"),
    Input("ls-y1", "value"),
    Input("ls-y2", "value"),
    Input("ls-y3", "value"),
]


def register_callbacks(app: Dash, body_id: str) -> None:
    register_body_callback(app, body_id, INPUTS, render_body, page="least_squares")
