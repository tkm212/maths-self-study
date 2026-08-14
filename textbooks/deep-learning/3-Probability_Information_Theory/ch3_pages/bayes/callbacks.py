"""Dash callbacks for the Bayes page."""

from __future__ import annotations

from dash import Dash, Input

from ch3_pages.bayes.content import render_body
from maths_self_study.dashboards.callbacks import register_body_callback

INPUTS = [
    Input("bayes-prior", "value"),
    Input("bayes-sens", "value"),
    Input("bayes-fpr", "value"),
    Input("bayes-chosen", "value"),
    Input("bayes-opened", "value"),
]


def register_callbacks(app: Dash, body_id: str) -> None:
    register_body_callback(app, body_id, INPUTS, render_body, page="bayes")
