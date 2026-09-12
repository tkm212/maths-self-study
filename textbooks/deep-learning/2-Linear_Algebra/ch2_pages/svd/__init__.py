"""SVD dashboard page."""

from __future__ import annotations

from ch2_pages.svd.callbacks import register_callbacks
from ch2_pages.svd.filters import build_filters
from maths_self_study.dashboards.page_factory import define_page
from maths_self_study.viz.textbooks.deep_learning.ch2.algorithms import SVD_LEAST_SQUARES as SVD_ALGORITHM
from maths_self_study.viz.textbooks.deep_learning.ch2.definitions import SVD as SVD_DEFINITIONS
from maths_self_study.viz.textbooks.deep_learning.ch2.theorems import SVD as SVD_THEOREMS

SvdPage = define_page(
    label="SVD",
    value="svd",
    title="SVD — every matrix has a geometry",
    caption="§2.8-2.9 — A = UΣVᵀ. Singular values are axis lengths of the unit ball's image.",
    summary=(
        "The singular value decomposition factors any matrix into rotation-"
        "scaling-rotation form. Singular values measure how much energy each "
        "direction carries through the map. We use SVD for compression, "
        "denoising, and as the backbone of PCA."
    ),
    methodology=[
        "Adjust the matrix entries and inspect singular values, the unit-circle image, and the least-squares fit.",
    ],
    algorithm=SVD_ALGORITHM,
    definitions=SVD_DEFINITIONS,
    theorems=SVD_THEOREMS,
    build_filters=build_filters,
    register_callbacks=register_callbacks,
)
