"""Cross-validation dashboard page."""

from __future__ import annotations

from ch7_pages.cross_validation.callbacks import register_callbacks
from ch7_pages.cross_validation.filters import build_filters
from maths_self_study.dashboards.page_factory import define_page
from maths_self_study.viz.textbooks.elements_of_statistical_learning.ch7.definitions import (
    CROSS_VALIDATION as CROSS_VALIDATION_DEFINITIONS,
)
from maths_self_study.viz.textbooks.elements_of_statistical_learning.ch7.theorems import (
    CROSS_VALIDATION as CROSS_VALIDATION_THEOREMS,
)

CrossValidationPage = define_page(
    label="Cross-validation",
    value="cross_validation",
    title="Cp, AIC, BIC, CV and bootstrap",
    caption="§7.5–7.11 — In-sample criteria and resampling estimates.",
    methodology=[
        "Cp, AIC and BIC penalise complexity to estimate test error.",
        "K-fold CV and the .632 bootstrap estimate generalisation directly.",
    ],
    definitions=CROSS_VALIDATION_DEFINITIONS,
    theorems=CROSS_VALIDATION_THEOREMS,
    build_filters=build_filters,
    register_callbacks=register_callbacks,
)
