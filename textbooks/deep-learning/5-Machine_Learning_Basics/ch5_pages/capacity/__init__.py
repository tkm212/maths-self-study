"""Capacity and overfitting dashboard page."""

from __future__ import annotations

from ch5_pages.capacity.callbacks import register_callbacks
from ch5_pages.capacity.filters import build_filters
from maths_self_study.dashboards.page_factory import define_page
from maths_self_study.viz.definitions.ch5 import CAPACITY as CAPACITY_DEFINITIONS

CapacityPage = define_page(
    label="Capacity",
    value="capacity",
    title="Model capacity and overfitting",
    caption="§5.2 — Polynomial degree controls how flexibly the model fits noisy training data.",
    methodology=[
        "Capacity is the model's ability to fit varied functions — higher-degree polynomials have more capacity.",
        "Underfitting: capacity too low; high training and test error.",
        "Overfitting: capacity too high; training error drops but test error rises.",
    ],
    definitions=CAPACITY_DEFINITIONS,
    build_filters=build_filters,
    register_callbacks=register_callbacks,
)
