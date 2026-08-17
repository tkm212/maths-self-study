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
    methodology=[
        "Compare L¹, L², and L∞ unit balls — the same numeric 'size' can look very different.",
        "Set x = (x₁, x₂) and read ‖x‖₁, ‖x‖₂ (and optionally ‖x‖∞) from the table.",
        "Toggle L∞ to see how the unit-ball geometry changes with the norm choice.",
        "Use cos(e₁, (1,1)) to connect inner products and angles (45° when cos = 1/√2).",
    ],
    build_filters=build_filters,
    register_callbacks=register_callbacks,
)
