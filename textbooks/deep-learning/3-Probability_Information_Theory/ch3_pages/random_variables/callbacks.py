"""Dash callbacks for the random variables page."""

from __future__ import annotations

from dash import Dash, Input

from ch3_pages.random_variables.content import render_body
from maths_self_study.dashboards.callbacks import register_body_callback

INPUTS = [
    Input("rv-j00", "value"),
    Input("rv-j01", "value"),
    Input("rv-j10", "value"),
    Input("rv-j11", "value"),
    Input("rv-p0", "value"),
    Input("rv-p1", "value"),
    Input("rv-p2", "value"),
    Input("rv-p3", "value"),
]


def register_callbacks(app: Dash, body_id: str) -> None:
    register_body_callback(app, body_id, INPUTS, render_body, page="random_variables")
