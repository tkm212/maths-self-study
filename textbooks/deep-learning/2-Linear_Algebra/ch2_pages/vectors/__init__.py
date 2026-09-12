"""Vectors & matrices dashboard page."""

from __future__ import annotations

from ch2_pages.vectors.callbacks import register_callbacks
from ch2_pages.vectors.filters import build_filters
from maths_self_study.dashboards.page_factory import define_page
from maths_self_study.viz.textbooks.deep_learning.ch2.definitions import VECTORS as VECTORS_DEFINITIONS

VectorsPage = define_page(
    label="Vectors & matrices",
    value="vectors",
    title="Linear maps as geometry",
    caption="§2.1-2.2 — A matrix A is a linear map x ↦ Ax. Columns of A are where the basis goes.",
    summary=(
        "Vectors and matrices represent data and linear transformations "
        "geometrically. A matrix maps input directions to output directions; "
        "composition chains transformations together. We use linear algebra "
        "as the language for everything from least squares to neural network "
        "layers."
    ),
    methodology=[
        "Each column of A shows where a basis vector lands under the linear map.",
        "Composition applies maps right-to-left: apply the inner map first, then the outer map.",
        "The inner product encodes angle between vectors — orthogonal when the inner product is zero.",
        "Elementary maps (rotation, shear) are building blocks; any linear map is their composition plus scaling.",
    ],
    definitions=VECTORS_DEFINITIONS,
    build_filters=build_filters,
    register_callbacks=register_callbacks,
)
