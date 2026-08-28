"""Separating hyperplanes dashboard page."""

from __future__ import annotations

from ch4_pages.separating_hyperplanes.callbacks import register_callbacks
from ch4_pages.separating_hyperplanes.filters import build_filters
from maths_self_study.dashboards.page_factory import define_page
from maths_self_study.viz.textbooks.elements_of_statistical_learning.ch4.definitions import (
    SEPARATING_HYPERPLANES as HYPERPLANE_DEFINITIONS,
)
from maths_self_study.viz.textbooks.elements_of_statistical_learning.ch4.theorems import (
    SEPARATING_HYPERPLANES as HYPERPLANE_THEOREMS,
)

SeparatingHyperplanesPage = define_page(
    label="Hyperplanes",
    value="separating_hyperplanes",
    title="Perceptron vs max-margin SVM",
    caption="§4.5 — Synthetic 2D separable data.",
    methodology=[
        "Perceptron finds a separator; SVM finds the unique max-margin one.",
        "Better generalisation from margin maximisation.",
    ],
    definitions=HYPERPLANE_DEFINITIONS,
    theorems=HYPERPLANE_THEOREMS,
    build_filters=build_filters,
    register_callbacks=register_callbacks,
)
