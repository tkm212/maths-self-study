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
        "Centre data: subtract the mean μ. Sample covariance Σ = (1/n) XᵀX captures spread and correlation.",
        "PCA finds orthonormal directions w₁, w₂, … that maximise variance; w₁ is the top eigenvector of Σ.",
        "Eigenvalues λᵢ of Σ equal variance along PCᵢ. Project: codes z = (X − μ) W; reconstruct with X̂ = μ + ZWᵀ.",
        "Keeping k components minimises reconstruction error; discarded variance = Σᵢ₌ₖ₊₁ λᵢ.",
    ],
    build_filters=build_filters,
    register_callbacks=register_callbacks,
)
