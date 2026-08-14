"""Norms dashboard page."""

from __future__ import annotations

from ch2_pages.norms.callbacks import register_callbacks
from ch2_pages.norms.filters import build_filters
from maths_self_study.dashboards.page_factory import define_page

NormsPage = define_page(
    label="Norms",
    value="norms",
    title="Norms as geometry",
    caption="§2.5 — ‖x‖ₚ unit balls: L² circle, L¹ diamond, L∞ square.",
    build_filters=build_filters,
    register_callbacks=register_callbacks,
)
