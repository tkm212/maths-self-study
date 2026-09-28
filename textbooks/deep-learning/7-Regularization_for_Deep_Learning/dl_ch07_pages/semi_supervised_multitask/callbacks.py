"""Dash callbacks for the semi-supervised / multitask page."""

from __future__ import annotations

from dash import Input

from dl_ch07_pages.semi_supervised_multitask.content import render_body
from maths_self_study.dashboards.callbacks import define_page_callbacks

INPUTS = [Input("ss-n-labeled", "value")]

register_callbacks = define_page_callbacks(
    render_body=render_body,
    inputs=INPUTS,
    page="semi_supervised_multitask",
)
