"""Eigendecomposition dashboard page."""

from __future__ import annotations

from ch2_pages._page_factory import page
from ch2_pages.eigendecomposition.callbacks import register_callbacks
from ch2_pages.eigendecomposition.filters import build_filters

EigendecompositionPage = page(
    label="Eigendecomposition",
    value="eigen",
    title="Eigendecomposition — invariant directions",
    caption="§2.7 — Av = λv. Symmetric A: A = QΛQᵀ.",
    build_filters=build_filters,
    register_callbacks=register_callbacks,
)
