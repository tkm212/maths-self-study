"""Dash callbacks for the tensors page."""

from __future__ import annotations

from dash import Dash, Input

from ch2_pages.tensors.content import render_body
from maths_self_study.dashboards.callbacks import register_body_callback
from maths_self_study.dashboards.components import tensor_callback_inputs
from maths_self_study.deep_learning import ch2_helpers as helpers

INPUTS = [
    *tensor_callback_inputs("tensor-grid", helpers.TENSOR_SHAPE),
    Input("tensor-axis", "value"),
    Input("tensor-slice", "value"),
]


def register_callbacks(app: Dash, body_id: str) -> None:
    register_body_callback(app, body_id, INPUTS, render_body, page="tensors")
