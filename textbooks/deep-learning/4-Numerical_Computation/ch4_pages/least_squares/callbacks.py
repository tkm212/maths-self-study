"""Dash callbacks for the least squares page."""

from __future__ import annotations

from ch4_pages.least_squares.content import render_body
from maths_self_study.dashboards.callbacks import define_page_callbacks, simplex_callback_inputs
from maths_self_study.dashboards.components import prob_simplex_ids

_TARGET_IDS = prob_simplex_ids("ls-y", [0, 1, 2, 3])

register_callbacks = define_page_callbacks(
    render_body=render_body,
    inputs=simplex_callback_inputs(_TARGET_IDS),
    page="least_squares",
)
