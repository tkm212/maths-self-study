"""Additive models dashboard page."""

from __future__ import annotations

from ch9_pages.additive_models.callbacks import register_callbacks
from ch9_pages.additive_models.filters import build_filters
from maths_self_study.dashboards.page_factory import define_page
from maths_self_study.viz.textbooks.elements_of_statistical_learning.ch9.algorithms import (
    ADDITIVE_MODELS as ADDITIVE_MODELS_ALGORITHM,
)
from maths_self_study.viz.textbooks.elements_of_statistical_learning.ch9.definitions import (
    ADDITIVE_MODELS as ADDITIVE_MODELS_DEFINITIONS,
)

AdditiveModelsPage = define_page(
    label="Additive models",
    value="additive_models",
    title="GAMs and backfitting",
    caption="§9.1 — Partial effects and spline smoothers.",
    algorithm=ADDITIVE_MODELS_ALGORITHM,
    definitions=ADDITIVE_MODELS_DEFINITIONS,
    build_filters=build_filters,
    register_callbacks=register_callbacks,
)
