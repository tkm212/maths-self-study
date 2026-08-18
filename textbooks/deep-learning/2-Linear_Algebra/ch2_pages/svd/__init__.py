"""SVD dashboard page."""

from __future__ import annotations

from ch2_pages.svd.callbacks import register_callbacks
from ch2_pages.svd.filters import build_filters
from maths_self_study.dashboards.page_factory import define_page

SvdPage = define_page(
    label="SVD",
    value="svd",
    title="SVD — every matrix has a geometry",
    caption="§2.8-2.9 — A = UΣVᵀ. Singular values are axis lengths of the unit ball's image.",
    methodology=[
        "Every A ∈ ℝᵐˣⁿ has A = UΣVᵀ with U, V orthogonal and Σ diagonal with σ₁ ≥ σ₂ ≥ … ≥ 0.",
        "NumPy: U, s, Vh = np.linalg.svd(A); reconstruct with U @ np.diag(s) @ Vh. Use pinv(A) for the Moore–Penrose inverse.",
        "σᵢ = √λᵢ(AᵀA). Geometrically, σᵢ are ellipse axis lengths for {Ax : ‖x‖₂ = 1}.",
        "Least squares: x = np.linalg.pinv(A) @ b minimises ‖Ax − b‖₂ when the system is overdetermined.",
    ],
    build_filters=build_filters,
    register_callbacks=register_callbacks,
)
