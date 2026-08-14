"""PCA dashboard page."""

from __future__ import annotations

from ch2_pages.pca.callbacks import register_callbacks
from ch2_pages.pca.filters import build_filters
from maths_self_study.dashboards.page_factory import define_page

PcaPage = define_page(
    label="PCA",
    value="pca",
    title="PCA — best low-dimensional view",
    caption="§2.12 — Orthogonal directions of maximal variance = eigenvectors of the covariance.",
    build_filters=build_filters,
    register_callbacks=register_callbacks,
)
