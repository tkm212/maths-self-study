"""Bias-variance dashboard page."""

from __future__ import annotations

from ch7_pages.bias_variance.callbacks import register_callbacks
from ch7_pages.bias_variance.filters import build_filters
from maths_self_study.dashboards.page_factory import define_page
from maths_self_study.viz.textbooks.elements_of_statistical_learning.ch7.definitions import (
    BIAS_VARIANCE as BIAS_VARIANCE_DEFINITIONS,
)
from maths_self_study.viz.textbooks.elements_of_statistical_learning.ch7.theorems import (
    BIAS_VARIANCE as BIAS_VARIANCE_THEOREMS,
)

BiasVariancePage = define_page(
    label="Bias-variance",
    value="bias_variance",
    title="Bias, variance and optimism",
    caption="§7.2–7.4 — Train/test error and the bias-variance tradeoff.",
    methodology=[
        "Squared-error decomposition at x₀: Err(x₀) = σ² + Bias²[ƒ̂(x₀)] + Var[ƒ̂(x₀)] — irreducible noise plus approximation and estimation error (§7.3).",
        "Train MSE falls as polynomial degree grows; test MSE is U-shaped — flexible models reduce bias but variance explodes when d is too large.",
        "Optimism = train error − test error; it grows with effective degrees of freedom because the same data are used to fit and evaluate (§7.4).",
        "The train/test gap on a TMDB feature mirrors the synthetic panels: compare the U-curve to the bias² + variance tradeoff.",
    ],
    definitions=BIAS_VARIANCE_DEFINITIONS,
    theorems=BIAS_VARIANCE_THEOREMS,
    build_filters=build_filters,
    register_callbacks=register_callbacks,
)
