"""Splines dashboard page."""

from __future__ import annotations

from ch5_pages.splines.callbacks import register_callbacks
from ch5_pages.splines.filters import build_filters
from maths_self_study.dashboards.page_factory import define_page

SplinesPage = define_page(
    label="Splines",
    value="splines",
    title="Basis expansions and splines",
    caption="§5.2 — Splines on a single TMDB feature.",
    methodology=[
        "More knots reduce bias but increase variance.",
        "Natural cubic splines add boundary constraints.",
    ],
    build_filters=build_filters,
    register_callbacks=register_callbacks,
)
