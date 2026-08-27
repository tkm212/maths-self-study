"""Vectors & matrices dashboard page."""

from __future__ import annotations

from ch2_pages.vectors.callbacks import register_callbacks
from ch2_pages.vectors.filters import build_filters
from maths_self_study.dashboards.page_factory import define_page
from maths_self_study.viz.definitions.ch2 import VECTORS as VECTORS_DEFINITIONS

VectorsPage = define_page(
    label="Vectors & matrices",
    value="vectors",
    title="Linear maps as geometry",
    caption="§2.1-2.2 — A matrix A is a linear map x ↦ Ax. Columns of A are where the basis goes.",
    methodology=[
        "A matrix A ∈ ℝᵐˣⁿ is a linear map x ↦ Ax. Column j is Aeⱼ — where the j-th basis vector lands.",
        "Composition applies maps right-to-left: (BA)x = B(Ax). Matrix multiply is associative but not commutative.",
        "The inner product xᵀy = Σᵢ xᵢyᵢ. In ℝ², xᵀy = ‖x‖₂ ‖y‖₂ cos θ — algebra encodes angle.",
        "Elementary maps (rotation R(θ), shear S(k)) are building blocks; any linear map is their composition plus scaling.",
    ],
    definitions=VECTORS_DEFINITIONS,
    build_filters=build_filters,
    register_callbacks=register_callbacks,
)
