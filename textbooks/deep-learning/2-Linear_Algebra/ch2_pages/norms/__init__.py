"""Norms dashboard page."""

from __future__ import annotations

from ch2_pages.norms.callbacks import register_callbacks
from ch2_pages.norms.filters import build_filters
from maths_self_study.dashboards.page_factory import define_page
from maths_self_study.viz.definitions.ch2 import NORMS as NORMS_DEFINITIONS
from maths_self_study.viz.theorems.ch2 import NORMS as NORMS_THEOREMS

NormsPage = define_page(
    label="Norms",
    value="norms",
    title="Norms as geometry",
    caption="§2.5 — ‖x‖ₚ unit balls: L² circle, L¹ diamond, L∞ square.",
    methodology=[
        "The Lᵖ norm is ‖x‖ₚ = (Σᵢ |xᵢ|ᵖ)^(1/p). Common cases: L¹ (Manhattan), L² (Euclidean), L∞ (max absolute entry).",
        "L²: ‖x‖₂ = √(xᵀx). The unit ball {x : ‖x‖₂ = 1} is a circle; L¹ gives a diamond, L∞ a square.",
        "All norms measure 'size' but weight coordinates differently — optimisation and regularisation depend on this choice.",
        "Cosine similarity cos θ = xᵀy / (‖x‖₂ ‖y‖₂) depends only on direction, not magnitude.",
    ],
    definitions=NORMS_DEFINITIONS,
    theorems=NORMS_THEOREMS,
    build_filters=build_filters,
    register_callbacks=register_callbacks,
)
