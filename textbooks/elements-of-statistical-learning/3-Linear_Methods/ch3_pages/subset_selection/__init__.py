"""Subset selection dashboard page."""

from __future__ import annotations

from ch3_pages.subset_selection.callbacks import register_callbacks
from ch3_pages.subset_selection.filters import build_filters
from maths_self_study.dashboards.page_factory import define_page
from maths_self_study.viz.textbooks.elements_of_statistical_learning.ch3.definitions import (
    SUBSET_SELECTION as SUBSET_SELECTION_DEFINITIONS,
)

SubsetSelectionPage = define_page(
    label="Subset selection",
    value="subset_selection",
    title="Forward stepwise selection",
    caption="§3.3 — Greedy feature selection on TMDB revenue.",
    methodology=[
        "At each step add the feature that most reduces held-out MSE.",
        "Entry order reveals marginal predictive power.",
    ],
    definitions=SUBSET_SELECTION_DEFINITIONS,
    build_filters=build_filters,
    register_callbacks=register_callbacks,
)
