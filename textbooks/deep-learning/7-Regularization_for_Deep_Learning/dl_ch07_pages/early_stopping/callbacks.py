"""Dash callbacks for the early stopping page."""

from __future__ import annotations

from dash import Input

from dl_ch07_pages.early_stopping.content import render_body
from maths_self_study.dashboards.callbacks import define_page_callbacks

INPUTS = [Input("es-stop-epoch", "value")]

register_callbacks = define_page_callbacks(
    render_body=render_body,
    inputs=INPUTS,
    page="early_stopping",
)
