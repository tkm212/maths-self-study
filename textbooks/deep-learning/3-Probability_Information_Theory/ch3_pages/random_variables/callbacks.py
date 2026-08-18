"""Dash callbacks for the random variables page."""

from __future__ import annotations

from dash import Dash, Input

from ch3_pages.random_variables.content import render_body
from maths_self_study.dashboards.callbacks import (
    register_body_callback,
    register_simplex_sync,
)

_JOINT_IDS = ["rv-j00", "rv-j01", "rv-j10", "rv-j11"]
_PMF_IDS = ["rv-p0", "rv-p1", "rv-p2", "rv-p3"]

INPUTS = [Input(id_, "value") for id_ in _JOINT_IDS + _PMF_IDS]


def register_callbacks(app: Dash, body_id: str) -> None:
    register_simplex_sync(app, _JOINT_IDS)
    register_simplex_sync(app, _PMF_IDS)
    register_body_callback(app, body_id, INPUTS, render_body, page="random_variables")
