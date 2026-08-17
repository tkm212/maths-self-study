"""Dash callbacks for the tensors page."""

from __future__ import annotations

from dash import Dash, Input

from ch2_pages.tensors.content import render_body
from maths_self_study.dashboards.callbacks import register_body_callback

INPUTS = [
    Input("tensor-a1", "value"),
    Input("tensor-a2", "value"),
    Input("tensor-b1", "value"),
    Input("tensor-b2", "value"),
    Input("tensor-b3", "value"),
    Input("tensor-axis", "value"),
    Input("tensor-slice", "value"),
]


def register_callbacks(app: Dash, body_id: str) -> None:
    register_body_callback(app, body_id, INPUTS, render_body, page="tensors")
