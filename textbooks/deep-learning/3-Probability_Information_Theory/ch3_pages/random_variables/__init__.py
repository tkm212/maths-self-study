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
        "Edit the 2×2 joint table P(rain, traffic); marginals appear by summing rows or columns.",
        "Form a conditional by slicing one variable and renormalising — P(A | B) = P(A, B) / P(B).",
        "Compute P(rain | heavy traffic) from the joint to see conditioning in action.",
        "Adjust a discrete PMF and read E[X] (centre of mass) and Var(X) (spread) from the moments plot.",
    ],
    build_filters=build_filters,
    register_callbacks=register_callbacks,
)
