"""Dash callbacks for the eigendecomposition page."""

from __future__ import annotations

from dash import Dash

from ch2_pages.eigendecomposition.content import render_body
from maths_self_study.dashboards.callbacks import register_body_callback
from maths_self_study.dashboards.components import matrix_callback_inputs

INPUTS = matrix_callback_inputs("cov-matrix")


def register_callbacks(app: Dash, body_id: str) -> None:
    register_body_callback(app, body_id, INPUTS, render_body, page="eigendecomposition")
