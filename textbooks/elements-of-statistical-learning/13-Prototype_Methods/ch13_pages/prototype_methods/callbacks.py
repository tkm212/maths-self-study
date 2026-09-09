"""Dash callbacks for prototype methods page."""

from __future__ import annotations

from ch13_pages.prototype_methods.content import render_body
from maths_self_study.dashboards.callbacks import define_page_callbacks

register_callbacks = define_page_callbacks(render_body=render_body, inputs=[], page="prototype_methods")
