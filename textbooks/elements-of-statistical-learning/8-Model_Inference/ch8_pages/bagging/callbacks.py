"""Dash callbacks for bagging page."""

from __future__ import annotations

from dash import Input

from ch8_pages.bagging.content import render_body
from maths_self_study.dashboards.callbacks import define_page_callbacks

INPUTS = [
    Input("bag-feat", "value"),
    Input("bag-degree", "value"),
    Input("bag-tree-depth", "value"),
    Input("bag-max-bags", "value"),
]

register_callbacks = define_page_callbacks(render_body=render_body, inputs=INPUTS, page="bagging")
