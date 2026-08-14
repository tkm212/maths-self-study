"""Vectors & matrices dashboard page."""

from __future__ import annotations

from ch2_pages.vectors.callbacks import register_callbacks
from ch2_pages.vectors.filters import build_filters
from maths_self_study.dashboards.page_factory import define_page

VectorsPage = define_page(
    label="Vectors & matrices",
    value="vectors",
    title="Linear maps as geometry",
    caption="§2.1-2.2 — A matrix A is a linear map x ↦ Ax. Columns of A are where the basis goes.",
    build_filters=build_filters,
    register_callbacks=register_callbacks,
)
