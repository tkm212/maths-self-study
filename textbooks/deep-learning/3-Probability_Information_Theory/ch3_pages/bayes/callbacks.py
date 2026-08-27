"""Dash callbacks for the Bayes page."""

from __future__ import annotations

from dash import Input

from ch3_pages.bayes.content import render_body
from maths_self_study.dashboards.callbacks import define_page_callbacks

INPUTS = [
    Input("bayes-prior", "value"),
    Input("bayes-sens", "value"),
    Input("bayes-fpr", "value"),
]

register_callbacks = define_page_callbacks(
    render_body=render_body,
    inputs=INPUTS,
    page="bayes",
)
