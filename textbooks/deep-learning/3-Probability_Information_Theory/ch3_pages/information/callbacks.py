"""Dash callbacks for the information theory page."""

from __future__ import annotations

from dash import Dash, Input

from ch3_pages.information.content import render_body
from maths_self_study.dashboards.callbacks import register_body_callback

INPUTS = [
    Input("info-p0", "value"),
    Input("info-p1", "value"),
    Input("info-p2", "value"),
    Input("info-p3", "value"),
    Input("info-q0", "value"),
    Input("info-q1", "value"),
    Input("info-q2", "value"),
    Input("info-q3", "value"),
]


def register_callbacks(app: Dash, body_id: str) -> None:
    register_body_callback(app, body_id, INPUTS, render_body, page="information")
