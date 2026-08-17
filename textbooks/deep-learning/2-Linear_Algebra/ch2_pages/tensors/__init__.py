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
    methodology=[
        "Set vectors a and b; the demo builds T[i, j, k] = a[i] b[j] c[k] — a rank-3 outer product.",
        "Pick a slice axis (i, j, or k) and index to view one 2D face of the tensor as a heatmap.",
        "Each slice is a scaled outer product a ⊗ b; rank-1 structure repeats across depth.",
        "Read shape, ndim, slice rank, and ‖T‖_F from the summary table.",
    ],
    build_filters=build_filters,
    register_callbacks=register_callbacks,
)
