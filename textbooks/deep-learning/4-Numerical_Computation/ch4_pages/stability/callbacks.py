"""Dash callbacks for the stability page."""

from __future__ import annotations

from dash import Dash, Input

from ch4_pages.stability.content import render_body
from maths_self_study.dashboards.callbacks import register_body_callback

INPUTS = [
    Input("stab-z0", "value"),
    Input("stab-z1", "value"),
    Input("stab-z2", "value"),
]


def register_callbacks(app: Dash, body_id: str) -> None:
    register_body_callback(app, body_id, INPUTS, render_body, page="stability")
