"""Linear least squares dashboard page."""

from __future__ import annotations

from ch4_pages.least_squares.callbacks import register_callbacks
from ch4_pages.least_squares.filters import build_filters
from maths_self_study.dashboards.page_factory import define_page
from maths_self_study.viz.definitions.ch4 import LEAST_SQUARES as LEAST_SQUARES_DEFINITIONS
from maths_self_study.viz.theorems.ch4 import LEAST_SQUARES as LEAST_SQUARES_THEOREMS

LeastSquaresPage = define_page(
    label="Least squares",
    value="least_squares",
    title="Linear least squares",
    caption="§4.5 — w* = (AᵀA)⁻¹Aᵀb minimizes ||Aw - b||₂; connects to Ch. 2 linear algebra.",
    methodology=[
        "Overdetermined system Aw ≈ b (more rows than columns) — no exact solution in general.",
        "Least squares: minimize ||Aw - b||₂² — normal equations AᵀAw* = Aᵀb.",
        "Solution w* = (AᵀA)⁻¹Aᵀb when AᵀA is invertible (full column rank).",
        "Same framework as linear regression; pseudoinverse A⁺ from Ch. 2 generalises the formula.",
    ],
    definitions=LEAST_SQUARES_DEFINITIONS,
    theorems=LEAST_SQUARES_THEOREMS,
    build_filters=build_filters,
    register_callbacks=register_callbacks,
)
