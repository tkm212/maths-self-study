"""Dash callbacks for principal components page."""

from __future__ import annotations

from dash import Input

from ch14_pages.principal_components.content import render_body
from maths_self_study.dashboards.callbacks import define_page_callbacks

INPUTS = [Input("pc-biplot-x", "value"), Input("pc-biplot-y", "value")]

register_callbacks = define_page_callbacks(
    render_body=render_body, inputs=INPUTS, page="principal_components"
)
