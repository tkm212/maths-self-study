"""Bagging ensemble page."""

from __future__ import annotations

from dl_ch07_pages.bagging.callbacks import register_callbacks
from dl_ch07_pages.bagging.filters import build_filters
from maths_self_study.dashboards.page_factory import define_page
from maths_self_study.viz.textbooks.deep_learning.ch7.algorithms import BAGGING as BAGGING_ALGORITHM
from maths_self_study.viz.textbooks.deep_learning.ch7.definitions import BAGGING as BAGGING_DEFINITIONS

BaggingPage = define_page(
    label="Bagging",
    value="bagging",
    title="Bagging and ensemble averaging",
    caption="§7.11 — Bootstrap models and average predictions to cut variance.",
    summary=(
        "Bagging trains multiple models on bootstrap resamples of the training "
        "set and averages their predictions. Errors that are not perfectly "
        "correlated partially cancel, often improving validation performance "
        "without changing the base architecture."
    ),
    methodology=BAGGING_ALGORITHM[1],
    definitions=BAGGING_DEFINITIONS,
    build_filters=build_filters,
    register_callbacks=register_callbacks,
)
