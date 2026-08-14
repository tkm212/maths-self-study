"""Dash callbacks for the Markov / structured models page."""

from __future__ import annotations

from dash import Dash, Input

from ch3_pages.markov.content import render_body
from maths_self_study.dashboards.callbacks import register_body_callback

INPUTS = [
    Input("mk-px1", "value"),
    Input("mk-t00", "value"),
    Input("mk-t10", "value"),
    Input("mk-u00", "value"),
    Input("mk-u10", "value"),
]


def register_callbacks(app: Dash, body_id: str) -> None:
    register_body_callback(app, body_id, INPUTS, render_body, page="markov")
