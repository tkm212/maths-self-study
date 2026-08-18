"""Dash callbacks for the Markov / structured models page."""

from __future__ import annotations

from dash import Dash, Input

from ch3_pages.markov.content import render_body
from maths_self_study.dashboards.callbacks import (
    register_body_callback,
    register_complement_pair,
)

INPUTS = [
    Input("mk-p-0", "value"),
    Input("mk-p-1", "value"),
    Input("mk-t0-0", "value"),
    Input("mk-t0-1", "value"),
    Input("mk-t1-0", "value"),
    Input("mk-t1-1", "value"),
    Input("mk-u0-0", "value"),
    Input("mk-u0-1", "value"),
    Input("mk-u1-0", "value"),
    Input("mk-u1-1", "value"),
]


def register_callbacks(app: Dash, body_id: str) -> None:
    for prefix in ("mk-p", "mk-t0", "mk-t1", "mk-u0", "mk-u1"):
        register_complement_pair(app, f"{prefix}-0", f"{prefix}-1")
    register_body_callback(app, body_id, INPUTS, render_body, page="markov")
