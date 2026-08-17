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
    methodology=[
        "Sample 2D data with adjustable stretch (sx, sy) and count; centre and compute the covariance.",
        "Fit PCA: principal axes are eigenvectors of the covariance, ordered by eigenvalue (variance).",
        "Compare raw data + axes, 1D projection onto PC1, and the explained-variance bar chart.",
        "Reconstruction error ‖X̂ − X‖ quantifies information lost when keeping fewer components.",
    ],
    build_filters=build_filters,
    register_callbacks=register_callbacks,
)
