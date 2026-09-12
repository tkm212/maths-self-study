"""PCA dashboard page."""

from __future__ import annotations

from ch2_pages.pca.callbacks import register_callbacks
from ch2_pages.pca.filters import build_filters
from maths_self_study.dashboards.page_factory import define_page
from maths_self_study.viz.textbooks.deep_learning.ch2.algorithms import PCA as PCA_ALGORITHM
from maths_self_study.viz.textbooks.deep_learning.ch2.definitions import PCA as PCA_DEFINITIONS
from maths_self_study.viz.textbooks.deep_learning.ch2.theorems import PCA as PCA_THEOREMS

PcaPage = define_page(
    label="PCA",
    value="pca",
    title="PCA — best low-dimensional view",
    caption="§2.12 — Orthogonal directions of maximal variance = eigenvectors of the covariance.",
    summary=(
        "Principal component analysis projects data onto the directions of "
        "maximum variance. The first few components capture most of the "
        "structure in a lower-dimensional view. We use it to visualise "
        "high-dimensional data and reduce dimensionality before modelling."
    ),
    methodology=[
        "Adjust the number of components and inspect variance explained versus reconstruction error.",
    ],
    algorithm=PCA_ALGORITHM,
    definitions=PCA_DEFINITIONS,
    theorems=PCA_THEOREMS,
    build_filters=build_filters,
    register_callbacks=register_callbacks,
)
