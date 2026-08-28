"""AdaBoost dashboard page."""

from __future__ import annotations

from ch10_pages.boosting.callbacks import register_callbacks
from ch10_pages.boosting.filters import build_filters
from maths_self_study.dashboards.page_factory import define_page
from maths_self_study.viz.textbooks.elements_of_statistical_learning.ch10.definitions import (
    BOOSTING as BOOSTING_DEFINITIONS,
)
from maths_self_study.viz.textbooks.elements_of_statistical_learning.ch10.theorems import (
    BOOSTING as BOOSTING_THEOREMS,
)

BoostingPage = define_page(
    label="Boosting",
    value="boosting",
    title="AdaBoost",
    caption="§10.1–10.4 — Stumps, training curves and margins.",
    methodology=[
        "AdaBoost reweights misclassified examples each round.",
        "Margin distribution shifts right as rounds increase.",
    ],
    definitions=BOOSTING_DEFINITIONS,
    theorems=BOOSTING_THEOREMS,
    build_filters=build_filters,
    register_callbacks=register_callbacks,
)
