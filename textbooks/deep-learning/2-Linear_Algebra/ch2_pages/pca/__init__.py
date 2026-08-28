"""PCA dashboard page."""

from __future__ import annotations

from ch2_pages.pca.callbacks import register_callbacks
from ch2_pages.pca.filters import build_filters
from maths_self_study.dashboards.page_factory import define_page
from maths_self_study.viz.textbooks.deep_learning.ch2.definitions import PCA as PCA_DEFINITIONS
from maths_self_study.viz.textbooks.deep_learning.ch2.theorems import PCA as PCA_THEOREMS

PcaPage = define_page(
    label="PCA",
    value="pca",
    title="PCA — best low-dimensional view",
    caption="§2.12 — Orthogonal directions of maximal variance = eigenvectors of the covariance.",
    methodology=[
        "Centre X to X_c = X − μ. Sample covariance Σ = X_cᵀX_c / (n − 1) is symmetric and captures spread.",
        "PCA directions are eigenvectors of Σ: λ, Q = np.linalg.eigh(Σ), sorted so λ₁ ≥ λ₂ ≥ …",
        "Project with Z = X_c @ Wᵀ (W = top k eigenvectors as rows); reconstruct with X̂ = Z @ W + μ.",
        "Same result from SVD: np.linalg.svd(X_c) gives PCs in Vh; truncating k components minimises reconstruction error.",
    ],
    definitions=PCA_DEFINITIONS,
    theorems=PCA_THEOREMS,
    build_filters=build_filters,
    register_callbacks=register_callbacks,
)
