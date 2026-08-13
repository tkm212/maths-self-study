"""SVD dashboard page."""

from __future__ import annotations

from ch2_pages._page_factory import page
from ch2_pages.svd.callbacks import register_callbacks
from ch2_pages.svd.filters import build_filters

SvdPage = page(
    label="SVD",
    value="svd",
    title="SVD — every matrix has a geometry",
    caption="§2.8-2.9 — A = UΣVᵀ. Singular values are axis lengths of the unit ball's image.",
    build_filters=build_filters,
    register_callbacks=register_callbacks,
)
