"""Eigendecomposition dashboard page."""

from __future__ import annotations

from ch2_pages.eigendecomposition.callbacks import register_callbacks
from ch2_pages.eigendecomposition.filters import build_filters
from maths_self_study.dashboards.page_factory import define_page
from maths_self_study.viz.textbooks.deep_learning.ch2.algorithms import (
    SYMMETRIC_EIGENDECOMPOSITION as EIGEN_ALGORITHM,
)
from maths_self_study.viz.textbooks.deep_learning.ch2.definitions import EIGEN as EIGEN_DEFINITIONS
from maths_self_study.viz.textbooks.deep_learning.ch2.theorems import EIGEN as EIGEN_THEOREMS

EigendecompositionPage = define_page(
    label="Eigendecomposition",
    value="eigen",
    title="Eigendecomposition — invariant directions",
    caption="§2.7 — Av = λv. Symmetric A: A = QΛQᵀ.",
    summary=(
        "Eigendecomposition finds directions a linear transformation stretches "
        "without rotating - the eigenvectors - and by how much - the "
        "eigenvalues. Symmetric matrices have real eigenpairs that reveal "
        "principal axes of variation. We use them to understand and simplify "
        "linear systems."
    ),
    methodology=[
        "Adjust matrix entries and inspect eigenvectors as invariant directions and the spectral reconstruction error.",
    ],
    algorithm=EIGEN_ALGORITHM,
    definitions=EIGEN_DEFINITIONS,
    theorems=EIGEN_THEOREMS,
    build_filters=build_filters,
    register_callbacks=register_callbacks,
)
