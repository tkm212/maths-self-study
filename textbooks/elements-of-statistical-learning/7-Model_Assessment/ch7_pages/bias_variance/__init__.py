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
        "Test error decomposes into bias², variance, and irreducible noise.",
        "Training error is optimistically biased; optimism grows with complexity.",
    ],
    definitions=BIAS_VARIANCE_DEFINITIONS,
    theorems=BIAS_VARIANCE_THEOREMS,
    build_filters=build_filters,
    register_callbacks=register_callbacks,
)
