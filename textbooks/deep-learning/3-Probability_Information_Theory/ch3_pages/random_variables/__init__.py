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
    methodology=[
        "A joint distribution P(X, Y) assigns probabilities to pairs; entries must be non-negative and sum to 1.",
        "Marginal: P(X = x) = Σ_y P(X = x, Y = y) — sum out the variable you don't care about.",
        "Conditional: P(X | Y = y) = P(X, Y = y) / P(Y = y) — restrict to one row/column and renormalise.",
        "Expectation E[X] = Σ x P(x); variance Var(X) = E[(X − E[X])²] = E[X²] − E[X]².",
    ],
    build_filters=build_filters,
    register_callbacks=register_callbacks,
)
