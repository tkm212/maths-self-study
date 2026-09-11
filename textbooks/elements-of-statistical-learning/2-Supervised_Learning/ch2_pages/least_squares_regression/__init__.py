"""Least squares dashboard page."""

from __future__ import annotations

from ch2_pages.least_squares_regression.callbacks import register_callbacks
from ch2_pages.least_squares_regression.filters import build_filters
from maths_self_study.dashboards.page_factory import define_page
from maths_self_study.viz.textbooks.elements_of_statistical_learning.ch2.definitions import (
    LEAST_SQUARES as LEAST_SQUARES_DEFINITIONS,
)
from maths_self_study.viz.textbooks.elements_of_statistical_learning.ch2.theorems import (
    LEAST_SQUARES as LEAST_SQUARES_THEOREMS,
)

LeastSquaresRegressionPage = define_page(
    label="Least squares",
    value="least_squares_regression",
    title="Closed-form OLS",
    caption="§2.3 — OLS on TMDB revenue with no tuning parameter.",
    summary=(
        "Ordinary least squares finds the best linear combination of predictors by "
        "minimising squared prediction error. It has a closed-form solution and every "
        "feature contributes at once. We use it as the simplest, most interpretable "
        "regression baseline - and as a reference point for more flexible methods."
    ),
    methodology=[
        "β̂ = argmin ||y − Xβ||² = (XᵀX)⁻¹Xᵀy — all features included, no shrinkage.",
        "Train MSE is optimistically biased; test MSE estimates generalisation error.",
    ],
    definitions=LEAST_SQUARES_DEFINITIONS,
    theorems=LEAST_SQUARES_THEOREMS,
    build_filters=build_filters,
    register_callbacks=register_callbacks,
)
