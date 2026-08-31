"""Dash callbacks for graphical models page."""

from __future__ import annotations

from ch17_pages.graphical_models.content import render_body
from maths_self_study.dashboards.callbacks import define_page_callbacks

register_callbacks = define_page_callbacks(render_body=render_body, inputs=[], page="graphical_models")
