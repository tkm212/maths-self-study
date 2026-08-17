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
        "Every A ∈ ℝᵐˣⁿ has a singular value decomposition A = UΣVᵀ with U, V orthogonal and Σ diagonal.",
        "Singular values σ₁ ≥ σ₂ ≥ … ≥ 0 are the diagonal entries of Σ — square roots of eigenvalues of AᵀA.",
        "Geometrically, σᵢ are the axis lengths of the ellipse {Ax : ‖x‖₂ = 1}; U gives output directions, V input.",
        "The Moore–Penrose pseudoinverse A⁺ = VΣ⁺Uᵀ solves min ‖Ax − b‖₂ via x = A⁺b (least squares when m > n).",
    ],
    build_filters=build_filters,
    register_callbacks=register_callbacks,
)
