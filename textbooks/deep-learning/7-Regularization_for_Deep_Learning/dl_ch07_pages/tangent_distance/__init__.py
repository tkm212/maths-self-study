"""Tangent distance page."""

from __future__ import annotations

from dl_ch07_pages.tangent_distance.callbacks import register_callbacks
from dl_ch07_pages.tangent_distance.filters import build_filters
from maths_self_study.dashboards.page_factory import define_page
from maths_self_study.viz.textbooks.deep_learning.ch7.definitions import (
    TANGENT_DISTANCE as TANGENT_DISTANCE_DEFINITIONS,
)

TangentDistancePage = define_page(
    label="Tangent distance",
    value="tangent_distance",
    title="Tangent distance and invariance",
    caption="§7.14 — Measure distance along known transformation directions.",
    summary=(
        "Tangent distance measures how far two points are after accounting for "
        "small transformations such as translation along a curve. Euclidean "
        "distance between on-manifold points can be large even when they differ "
        "only by a nuisance translation; tangent distance is much smaller."
    ),
    methodology=[
        "Pick two points on y = sin(2x) related by horizontal shift.",
        "Compute Euclidean distance in the plane.",
        "Project onto the translation tangent and measure residual distance.",
    ],
    definitions=TANGENT_DISTANCE_DEFINITIONS,
    build_filters=build_filters,
    register_callbacks=register_callbacks,
)
