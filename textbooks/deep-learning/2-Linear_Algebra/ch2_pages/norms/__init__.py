"""Norms dashboard page."""

from __future__ import annotations

from ch2_pages.norms.callbacks import register_callbacks
from ch2_pages.norms.filters import build_filters
from maths_self_study.dashboards.page_factory import define_page
from maths_self_study.viz.textbooks.deep_learning.ch2.definitions import NORMS as NORMS_DEFINITIONS
from maths_self_study.viz.textbooks.deep_learning.ch2.theorems import NORMS as NORMS_THEOREMS

NormsPage = define_page(
    label="Norms",
    value="norms",
    title="Norms as geometry",
    caption="§2.5 — ‖x‖ₚ unit balls: L² circle, L¹ diamond, L∞ square.",
    summary=(
        "Norms measure the size of vectors and quantify how far apart points "
        "are. Different norms induce different geometries and promote "
        "different kinds of regularisation. We use them to define loss "
        "functions, constraints, and distance-based methods throughout "
        "machine learning."
    ),
    methodology=[
        "Compare L1, L2, and L infinity unit balls — each norm weights coordinates differently.",
        "L2 gives a circle; L1 a diamond; L infinity a square in two dimensions.",
        "Cosine similarity depends only on direction, not vector magnitude.",
    ],
    definitions=NORMS_DEFINITIONS,
    theorems=NORMS_THEOREMS,
    build_filters=build_filters,
    register_callbacks=register_callbacks,
)
