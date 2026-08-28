"""Eigendecomposition dashboard page."""

from __future__ import annotations

from ch2_pages.eigendecomposition.callbacks import register_callbacks
from ch2_pages.eigendecomposition.filters import build_filters
from maths_self_study.dashboards.page_factory import define_page
from maths_self_study.viz.textbooks.deep_learning.ch2.definitions import EIGEN as EIGEN_DEFINITIONS
from maths_self_study.viz.textbooks.deep_learning.ch2.theorems import EIGEN as EIGEN_THEOREMS

EigendecompositionPage = define_page(
    label="Eigendecomposition",
    value="eigen",
    title="Eigendecomposition — invariant directions",
    caption="§2.7 — Av = λv. Symmetric A: A = QΛQᵀ.",
    methodology=[
        "An eigenpair (λ, v) satisfies Av = λv with v ≠ 0. v is an invariant direction; λ is the stretch factor along it.",
        "For symmetric A, all eigenvalues are real and eigenvectors are orthogonal (spectral theorem).",
        "Spectral decomposition: A = QΛQᵀ — Q's columns are orthonormal eigenvectors, Λ is diagonal of eigenvalues.",
        "This page symmetrises A, then calls np.linalg.eigh(A): LAPACK reduces A to tridiagonal form (Householder), solves for λ and Q, and verifies A ≈ QΛQᵀ.",
    ],
    definitions=EIGEN_DEFINITIONS,
    theorems=EIGEN_THEOREMS,
    build_filters=build_filters,
    register_callbacks=register_callbacks,
)
