"""Random variables dashboard page."""

from __future__ import annotations

from ch3_pages._page_factory import page
from ch3_pages.random_variables.callbacks import register_callbacks
from ch3_pages.random_variables.filters import build_filters

RandomVariablesPage = page(
    label="Random variables",
    value="rv",
    title="Probability as bookkeeping",
    caption="§3.2-3.8 — Joint → marginals (sum out) → conditionals (slice and renormalise).",
    build_filters=build_filters,
    register_callbacks=register_callbacks,
)
