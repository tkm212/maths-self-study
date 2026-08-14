"""Dash callbacks for the norms page."""

from __future__ import annotations

from dash import Dash, Input

from ch2_pages.norms.content import render_body
from maths_self_study.dashboards.callbacks import register_body_callback

INPUTS = [
    Input("norm-x1", "value"),
    Input("norm-x2", "value"),
    Input("norm-inf", "value"),
]


def register_callbacks(app: Dash, body_id: str) -> None:
    register_body_callback(app, body_id, INPUTS, render_body, page="norms")
