"""Lasso dashboard page."""

from __future__ import annotations

from ch3_pages.lasso.callbacks import register_callbacks
from ch3_pages.lasso.filters import build_filters
from maths_self_study.dashboards.page_factory import define_page
from maths_self_study.viz.textbooks.elements_of_statistical_learning.ch3.definitions import (
    LASSO as LASSO_DEFINITIONS,
)
from maths_self_study.viz.textbooks.elements_of_statistical_learning.ch3.theorems import (
    LASSO as LASSO_THEOREMS,
)

LassoPage = define_page(
    label="Lasso",
    value="lasso",
    title="L1 paths and feature selection",
    caption="§3.4 / §3.8 — Lasso on TMDB revenue.",
    methodology=[
        "L1 constraint creates exact zeros.",
        "Path entry order = marginal predictive importance.",
    ],
    definitions=LASSO_DEFINITIONS,
    theorems=LASSO_THEOREMS,
    build_filters=build_filters,
    register_callbacks=register_callbacks,
)
