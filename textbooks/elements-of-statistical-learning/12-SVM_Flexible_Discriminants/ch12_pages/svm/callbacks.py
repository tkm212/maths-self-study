"""Dash callbacks for SVM page."""

from __future__ import annotations

from dash import Input

from ch12_pages.svm.content import render_body
from maths_self_study.dashboards.callbacks import define_page_callbacks

INPUTS = [Input("svm-cost-kernel", "value"), Input("svm-kernel-c", "value")]

register_callbacks = define_page_callbacks(render_body=render_body, inputs=INPUTS, page="svm")
