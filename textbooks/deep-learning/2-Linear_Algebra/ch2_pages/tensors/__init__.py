"""Tensors dashboard page."""

from __future__ import annotations

from ch2_pages.tensors.callbacks import register_callbacks
from ch2_pages.tensors.filters import build_filters
from maths_self_study.dashboards.page_factory import define_page

TensorsPage = define_page(
    label="Tensors",
    value="tensors",
    title="Tensors — rank and indexing",
    caption="§2.3 — Scalars (0D), vectors (1D), matrices (2D). Outer products and slices build higher rank.",
    build_filters=build_filters,
    register_callbacks=register_callbacks,
)
