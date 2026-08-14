"""Dash callbacks for the distributions page."""

from __future__ import annotations

from dash import Dash, Input

from ch3_pages.distributions.content import render_body
from maths_self_study.dashboards.callbacks import register_body_callback

INPUTS = [
    Input("dist-c0", "value"),
    Input("dist-c1", "value"),
    Input("dist-c2", "value"),
    Input("dist-c3", "value"),
    Input("dist-cov-matrix", "value"),
]


def register_callbacks(app: Dash, body_id: str) -> None:
    register_body_callback(app, body_id, INPUTS, render_body, page="distributions")
