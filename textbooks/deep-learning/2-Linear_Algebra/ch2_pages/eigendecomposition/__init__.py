"""Eigendecomposition dashboard page."""

from __future__ import annotations

from ch2_pages.eigendecomposition.callbacks import register_callbacks
from ch2_pages.eigendecomposition.filters import build_filters
from maths_self_study.dashboards.page_factory import define_page

EigendecompositionPage = define_page(
    label="Eigendecomposition",
    value="eigen",
    title="Eigendecomposition — invariant directions",
    caption="§2.7 — Av = λv. Symmetric A: A = QΛQᵀ.",
    methodology=[
        "Edit matrix A; the symmetric part (A + Aᵀ)/2 is used so eigenvalues stay real.",
        "Plot the deformed grid with eigenvector arrows — directions that only stretch, not rotate.",
        "Eigenvalue λᵢ is the stretch factor along eigenvector vᵢ; Avᵢ = λᵢvᵢ.",
        "Check ‖A − QΛQᵀ‖ in the table to verify the spectral reconstruction.",
    ],
    build_filters=build_filters,
    register_callbacks=register_callbacks,
)
