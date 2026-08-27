"""Dash callbacks for the eigendecomposition page."""

from __future__ import annotations

from ch2_pages.eigendecomposition.content import render_body
from maths_self_study.dashboards.callbacks import define_page_callbacks
from maths_self_study.dashboards.components import matrix_callback_inputs

INPUTS = matrix_callback_inputs("cov-matrix")

register_callbacks = define_page_callbacks(
    render_body=render_body,
    inputs=INPUTS,
    page="eigendecomposition",
)
