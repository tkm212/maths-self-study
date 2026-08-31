"""Dash callbacks for the least_squares_regression page."""

from __future__ import annotations

from ch2_pages.least_squares_regression.content import render_body
from maths_self_study.dashboards.callbacks import TAB_TRIGGER, define_page_callbacks

register_callbacks = define_page_callbacks(
    render_body=render_body,
    inputs=[TAB_TRIGGER],
    page="least_squares_regression",
)
