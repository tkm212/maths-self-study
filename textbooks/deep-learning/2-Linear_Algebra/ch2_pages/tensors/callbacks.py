"""Dash callbacks for the tensors page."""

from __future__ import annotations

from dash import Input

from ch2_pages.tensors.content import render_body
from maths_self_study.dashboards.callbacks import define_page_callbacks
from maths_self_study.dashboards.components import tensor_callback_inputs
from maths_self_study.demos.deep_learning import ch2 as helpers

INPUTS = [
    *tensor_callback_inputs("tensor-grid", helpers.TENSOR_SHAPE),
    Input("tensor-axis", "value"),
    Input("tensor-slice", "value"),
]

register_callbacks = define_page_callbacks(
    render_body=render_body,
    inputs=INPUTS,
    page="tensors",
)
