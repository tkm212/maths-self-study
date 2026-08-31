"""Dash callbacks for the subset_selection page."""

from __future__ import annotations

from ch3_pages.subset_selection.content import render_body
from maths_self_study.dashboards.callbacks import TAB_TRIGGER, define_page_callbacks

register_callbacks = define_page_callbacks(
    render_body=render_body,
    inputs=[TAB_TRIGGER],
    page="subset_selection",
)
