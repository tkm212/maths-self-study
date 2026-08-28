"""Bagging dashboard page."""

from __future__ import annotations

from ch8_pages.bagging.callbacks import register_callbacks
from ch8_pages.bagging.filters import build_filters
from maths_self_study.dashboards.page_factory import define_page
from maths_self_study.viz.textbooks.elements_of_statistical_learning.ch8.definitions import (
    BAGGING as BAGGING_DEFINITIONS,
)
from maths_self_study.viz.textbooks.elements_of_statistical_learning.ch8.theorems import (
    BAGGING as BAGGING_THEOREMS,
)

BaggingPage = define_page(
    label="Bagging",
    value="bagging",
    title="Bootstrap and bagging",
    caption="§8.2, §8.7 — Bootstrap confidence bands and bagged trees.",
    methodology=[
        "Bootstrap resamples estimate sampling variability.",
        "Bagging averages high-variance learners to reduce variance.",
    ],
    definitions=BAGGING_DEFINITIONS,
    theorems=BAGGING_THEOREMS,
    build_filters=build_filters,
    register_callbacks=register_callbacks,
)
