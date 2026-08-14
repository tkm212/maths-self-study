"""Random variables dashboard page."""

from __future__ import annotations

from ch3_pages.random_variables.callbacks import register_callbacks
from ch3_pages.random_variables.filters import build_filters
from maths_self_study.dashboards.page_factory import define_page

RandomVariablesPage = define_page(
    label="Random variables",
    value="rv",
    title="Probability as bookkeeping",
    caption="§3.2-3.8 — Joint → marginals (sum out) → conditionals (slice and renormalise).",
    build_filters=build_filters,
    register_callbacks=register_callbacks,
)
