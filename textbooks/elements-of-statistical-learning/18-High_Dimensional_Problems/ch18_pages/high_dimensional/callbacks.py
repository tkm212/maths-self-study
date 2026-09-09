"""Dash callbacks for high-dimensional page."""

from __future__ import annotations

from ch18_pages.high_dimensional.content import render_body
from maths_self_study.dashboards.callbacks import define_page_callbacks

register_callbacks = define_page_callbacks(render_body=render_body, inputs=[], page="high_dimensional")
