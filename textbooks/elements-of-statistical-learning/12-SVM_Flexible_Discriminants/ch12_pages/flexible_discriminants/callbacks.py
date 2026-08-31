"""Dash callbacks for flexible discriminants page."""

from __future__ import annotations

from ch12_pages.flexible_discriminants.content import render_body
from maths_self_study.dashboards.callbacks import define_page_callbacks

register_callbacks = define_page_callbacks(render_body=render_body, inputs=[], page="flexible_discriminants")
