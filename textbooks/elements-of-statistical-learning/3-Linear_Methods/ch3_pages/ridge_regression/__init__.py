"""Ridge regression dashboard page."""

from __future__ import annotations

from ch3_pages.ridge_regression.callbacks import register_callbacks
from ch3_pages.ridge_regression.filters import build_filters
from maths_self_study.dashboards.page_factory import define_page
from maths_self_study.viz.textbooks.elements_of_statistical_learning.ch3.definitions import (
    RIDGE as RIDGE_DEFINITIONS,
)
from maths_self_study.viz.textbooks.elements_of_statistical_learning.ch3.theorems import (
    RIDGE as RIDGE_THEOREMS,
)

RidgeRegressionPage = define_page(
    label="Ridge",
    value="ridge_regression",
    title="L2 shrinkage paths",
    caption="§3.4 — Ridge regression on scaled TMDB features.",
    methodology=[
        "Small α ≈ OLS; large α → heavy shrinkage and high bias.",
        "No coefficient reaches exactly zero.",
    ],
    definitions=RIDGE_DEFINITIONS,
    theorems=RIDGE_THEOREMS,
    build_filters=build_filters,
    register_callbacks=register_callbacks,
)
