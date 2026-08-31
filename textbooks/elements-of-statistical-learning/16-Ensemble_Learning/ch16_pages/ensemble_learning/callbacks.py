"""Dash callbacks for ensemble learning page."""

from __future__ import annotations

from ch16_pages.ensemble_learning.content import render_body
from maths_self_study.dashboards.callbacks import define_page_callbacks

register_callbacks = define_page_callbacks(render_body=render_body, inputs=[], page="ensemble_learning")
