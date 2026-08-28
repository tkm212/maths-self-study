"""Lasso dashboard page."""

from __future__ import annotations

from ch3_pages.lasso.callbacks import register_callbacks
from ch3_pages.lasso.filters import build_filters
from maths_self_study.dashboards.page_factory import define_page

LassoPage = define_page(
    label="Lasso",
    value="lasso",
    title="L1 paths and feature selection",
    caption="§3.4 / §3.8 — Lasso on TMDB revenue.",
    methodology=[
        "L1 constraint creates exact zeros.",
        "Path entry order = marginal predictive importance.",
    ],
    build_filters=build_filters,
    register_callbacks=register_callbacks,
)
