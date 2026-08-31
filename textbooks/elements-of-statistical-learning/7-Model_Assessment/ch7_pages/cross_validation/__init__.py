"""Cross-validation dashboard page."""

from __future__ import annotations

from ch7_pages.cross_validation.callbacks import register_callbacks
from ch7_pages.cross_validation.filters import build_filters
from maths_self_study.dashboards.page_factory import define_page
from maths_self_study.viz.textbooks.elements_of_statistical_learning.ch7.algorithms import (
    CROSS_VALIDATION as CROSS_VALIDATION_ALGORITHM,
)
from maths_self_study.viz.textbooks.elements_of_statistical_learning.ch7.definitions import (
    CROSS_VALIDATION as CROSS_VALIDATION_DEFINITIONS,
)

CrossValidationPage = define_page(
    label="Cross-validation",
    value="cross_validation",
    title="Cp, AIC, BIC, CV and bootstrap",
    caption="§7.5–7.11 — In-sample criteria and resampling estimates.",
    methodology=[
        r"In-sample criteria penalise complexity without refitting: $C_p$, AIC, and BIC (§7.5–7.7).",
        r"BIC adds the strongest penalty ($\log N$ vs $2$); bootstrap .632 blends train and OOB error (§7.11).",
    ],
    algorithm=CROSS_VALIDATION_ALGORITHM,
    definitions=CROSS_VALIDATION_DEFINITIONS,
    build_filters=build_filters,
    register_callbacks=register_callbacks,
)
