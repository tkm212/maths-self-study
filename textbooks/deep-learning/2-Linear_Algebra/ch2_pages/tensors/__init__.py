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
        "A tensor is an n-dimensional array: scalar (0D), vector (1D), matrix (2D), then higher orders.",
        "Indexing T[i, j, k, …] picks one element along each mode; slicing fixes indices to get a lower-rank view.",
        "Outer product: (a ⊗ b)ᵢⱼ = aᵢ bⱼ — two vectors produce a rank-1 matrix.",
        "Full tensor product: T[i, j, k] = a[i] b[j] c[k] builds rank 3 from three vectors; each slice is a scaled outer product.",
    ],
    build_filters=build_filters,
    register_callbacks=register_callbacks,
)
