"""Splines dashboard page."""

from __future__ import annotations

from ch5_pages.splines.callbacks import register_callbacks
from ch5_pages.splines.filters import build_filters
from maths_self_study.dashboards.page_factory import define_page
from maths_self_study.viz.textbooks.elements_of_statistical_learning.ch5.definitions import (
    SPLINES as SPLINES_DEFINITIONS,
)

SplinesPage = define_page(
    label="Splines",
    value="splines",
    title="Basis expansions and splines",
    caption="§5.2 — Splines on a single TMDB feature.",
    summary=(
        "Splines model curved relationships by stitching together simple polynomials "
        "between knot points. Each piece is low-degree, so the overall curve can bend "
        "flexibly without wild oscillations. We use them when a straight line is too "
        "rigid but we still want smooth, locally controlled flexibility."
    ),
    methodology=[
        "More knots reduce bias but increase variance.",
        "Natural cubic splines add boundary constraints.",
    ],
    definitions=SPLINES_DEFINITIONS,
    build_filters=build_filters,
    register_callbacks=register_callbacks,
)
