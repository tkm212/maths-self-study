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
        "Edit the 2×2 map A and watch the unit circle map to an ellipse — image of ‖x‖₂ = 1.",
        "Read singular values sᵢ as the ellipse axis lengths; they generalise eigenvalues to non-square A.",
        "Set the right-hand side b for the overdetermined system Ax ≈ b.",
        "The table reports the least-squares solution x = A⁺b, residual ‖Ax − b‖₂, and σ(A).",
    ],
    build_filters=build_filters,
    register_callbacks=register_callbacks,
)
