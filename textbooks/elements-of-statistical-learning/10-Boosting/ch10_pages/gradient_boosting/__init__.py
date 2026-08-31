"""Gradient boosting dashboard page."""

from __future__ import annotations

from ch10_pages.gradient_boosting.callbacks import register_callbacks
from ch10_pages.gradient_boosting.filters import build_filters
from maths_self_study.dashboards.page_factory import define_page
from maths_self_study.viz.textbooks.elements_of_statistical_learning.ch10.algorithms import (
    GRADIENT_BOOSTING as GRADIENT_BOOSTING_ALGORITHM,
)
from maths_self_study.viz.textbooks.elements_of_statistical_learning.ch10.definitions import (
    GRADIENT_BOOSTING as GRADIENT_BOOSTING_DEFINITIONS,
)

GradientBoostingPage = define_page(
    label="Gradient boosting",
    value="gradient_boosting",
    title="GBM and shrinkage",
    caption="§10.9–10.13 — Trees, learning rate and variable importance.",
    algorithm=GRADIENT_BOOSTING_ALGORITHM,
    definitions=GRADIENT_BOOSTING_DEFINITIONS,
    build_filters=build_filters,
    register_callbacks=register_callbacks,
)
