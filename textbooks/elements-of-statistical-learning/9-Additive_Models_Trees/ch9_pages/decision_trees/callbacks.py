"""Dash callbacks for decision trees page."""

from __future__ import annotations

from dash import Input

from ch9_pages.decision_trees.content import render_body
from maths_self_study.dashboards.callbacks import define_page_callbacks

INPUTS = [Input("tree-max-depth", "value")]

register_callbacks = define_page_callbacks(render_body=render_body, inputs=INPUTS, page="decision_trees")
