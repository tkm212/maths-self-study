"""Splines dashboard page."""

from __future__ import annotations

from ch5_pages.splines.callbacks import register_callbacks
from ch5_pages.splines.filters import build_filters
from maths_self_study.dashboards.page_factory import define_page
from maths_self_study.viz.textbooks.elements_of_statistical_learning.ch5.definitions import (
    SPLINES as SPLINES_DEFINITIONS,
)
from maths_self_study.viz.textbooks.elements_of_statistical_learning.ch5.theorems import (
    SPLINES as SPLINES_THEOREMS,
)

SplinesPage = define_page(
    label="Splines",
    value="splines",
    title="Basis expansions and splines",
    caption="§5.2 — Splines on a single TMDB feature.",
    methodology=[
        "More knots reduce bias but increase variance.",
        "Natural cubic splines add boundary constraints.",
    ],
    definitions=SPLINES_DEFINITIONS,
    theorems=SPLINES_THEOREMS,
    build_filters=build_filters,
    register_callbacks=register_callbacks,
)
