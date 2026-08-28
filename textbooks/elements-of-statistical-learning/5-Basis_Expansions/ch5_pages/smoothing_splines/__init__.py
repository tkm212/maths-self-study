"""Smoothing splines dashboard page."""

from __future__ import annotations

from ch5_pages.smoothing_splines.callbacks import register_callbacks
from ch5_pages.smoothing_splines.filters import build_filters
from maths_self_study.dashboards.page_factory import define_page

SmoothingSplinesPage = define_page(
    label="Smoothing splines",
    value="smoothing_splines",
    title="Penalised roughness",
    caption="§5.4 — Smoothing splines and GCV on TMDB budget → revenue.",
    methodology=[
        "Effective df = tr(Sλ) controls flexibility.",
        "GCV selects λ without a held-out set.",
    ],
    build_filters=build_filters,
    register_callbacks=register_callbacks,
)
