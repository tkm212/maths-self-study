"""Additive models dashboard page."""

from __future__ import annotations

from ch9_pages.additive_models.callbacks import register_callbacks
from ch9_pages.additive_models.filters import build_filters
from maths_self_study.dashboards.page_factory import define_page
from maths_self_study.viz.textbooks.elements_of_statistical_learning.ch9.definitions import (
    ADDITIVE_MODELS as ADDITIVE_MODELS_DEFINITIONS,
)
from maths_self_study.viz.textbooks.elements_of_statistical_learning.ch9.theorems import (
    ADDITIVE_MODELS as ADDITIVE_MODELS_THEOREMS,
)

AdditiveModelsPage = define_page(
    label="Additive models",
    value="additive_models",
    title="GAMs and backfitting",
    caption="§9.1 — Partial effects and spline smoothers.",
    methodology=[
        "Each feature contributes an additive smooth term f_j(x_j).",
        "Backfitting cycles through partial residuals until convergence.",
    ],
    definitions=ADDITIVE_MODELS_DEFINITIONS,
    theorems=ADDITIVE_MODELS_THEOREMS,
    build_filters=build_filters,
    register_callbacks=register_callbacks,
)
