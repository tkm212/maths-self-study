"""AdaBoost dashboard page."""

from __future__ import annotations

from ch10_pages.boosting.callbacks import register_callbacks
from ch10_pages.boosting.filters import build_filters
from maths_self_study.dashboards.page_factory import define_page
from maths_self_study.viz.textbooks.elements_of_statistical_learning.ch10.algorithms import (
    BOOSTING as BOOSTING_ALGORITHM,
)
from maths_self_study.viz.textbooks.elements_of_statistical_learning.ch10.definitions import (
    BOOSTING as BOOSTING_DEFINITIONS,
)

BoostingPage = define_page(
    label="Boosting",
    value="boosting",
    title="AdaBoost",
    caption="§10.1–10.4 — Stumps, training curves and margins.",
    methodology=[
        r"Margin $y \cdot f(x)$ measures classification confidence — its distribution shifts right as rounds increase (§10.4).",
    ],
    algorithm=BOOSTING_ALGORITHM,
    definitions=BOOSTING_DEFINITIONS,
    build_filters=build_filters,
    register_callbacks=register_callbacks,
)
