"""Dash callbacks for the information theory page."""

from __future__ import annotations

from dash import Dash, Input

from ch3_pages.information.content import render_body
from maths_self_study.dashboards.callbacks import register_body_callback, register_simplex_sync

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

_P_IDS = ["info-p0", "info-p1", "info-p2", "info-p3"]
_Q_IDS = ["info-q0", "info-q1", "info-q2", "info-q3"]


def register_callbacks(app: Dash, body_id: str) -> None:
    register_simplex_sync(app, _P_IDS)
    register_simplex_sync(app, _Q_IDS)
    register_body_callback(app, body_id, INPUTS, render_body, page="information")
