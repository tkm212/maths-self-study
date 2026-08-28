"""Hyperplanes dashboard page."""

from __future__ import annotations

from ch4_pages.separating_hyperplanes.callbacks import register_callbacks
from ch4_pages.separating_hyperplanes.filters import build_filters
from maths_self_study.dashboards.page_factory import define_page

SeparatingHyperplanesPage = define_page(
    label="Hyperplanes",
    value="separating_hyperplanes",
    title="Perceptron vs max-margin SVM",
    caption="§4.5 — Synthetic 2D separable data.",
    methodology=[
        "Perceptron finds a separator; SVM finds the unique max-margin one.",
        "Better generalisation from margin maximisation.",
    ],
    build_filters=build_filters,
    register_callbacks=register_callbacks,
)
