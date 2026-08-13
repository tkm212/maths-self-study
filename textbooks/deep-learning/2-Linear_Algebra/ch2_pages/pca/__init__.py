"""PCA dashboard page."""

from __future__ import annotations

from ch2_pages._page_factory import page
from ch2_pages.pca.callbacks import register_callbacks
from ch2_pages.pca.filters import build_filters

PcaPage = page(
    label="PCA",
    value="pca",
    title="PCA — best low-dimensional view",
    caption="§2.12 — Orthogonal directions of maximal variance = eigenvectors of the covariance.",
    build_filters=build_filters,
    register_callbacks=register_callbacks,
)
