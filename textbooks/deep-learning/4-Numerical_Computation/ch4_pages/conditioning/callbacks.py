"""Dash callbacks for the conditioning page."""

from __future__ import annotations

from dash import Dash, Input

from ch4_pages.conditioning.content import render_body
from maths_self_study.dashboards.callbacks import register_body_callback

INPUTS = [
    Input("cond-kappa", "value"),
    Input("cond-delta", "value"),
]


def register_callbacks(app: Dash, body_id: str) -> None:
    register_body_callback(app, body_id, INPUTS, render_body, page="conditioning")
