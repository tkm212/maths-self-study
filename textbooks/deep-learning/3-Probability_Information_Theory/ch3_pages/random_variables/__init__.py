"""Random variables dashboard page."""

from __future__ import annotations

from ch3_pages.random_variables.callbacks import register_callbacks
from ch3_pages.random_variables.filters import build_filters
from maths_self_study.dashboards.page_factory import define_page
from maths_self_study.viz.textbooks.deep_learning.ch3.definitions import (
    RANDOM_VARIABLES as RANDOM_VARIABLES_DEFINITIONS,
)
from maths_self_study.viz.textbooks.deep_learning.ch3.theorems import RANDOM_VARIABLES as RANDOM_VARIABLES_THEOREMS

RandomVariablesPage = define_page(
    label="Random variables",
    value="rv",
    title="Probability as bookkeeping",
    caption="§3.2-3.8 — Joint → marginals (sum out) → conditionals (slice and renormalise).",
    summary=(
        "Random variables formalise uncertainty with probability "
        "distributions over outcomes. Joint distributions encode "
        "dependencies; marginals and conditionals let us query what we know "
        "and what remains uncertain. We use this framework to reason "
        "precisely about learning from noisy data."
    ),
    methodology=[
        "Adjust the joint table and inspect marginals, conditionals, and moments in the panels below.",
        "Marginals sum out one variable; conditionals renormalise a slice of the joint table.",
    ],
    definitions=RANDOM_VARIABLES_DEFINITIONS,
    theorems=RANDOM_VARIABLES_THEOREMS,
    build_filters=build_filters,
    register_callbacks=register_callbacks,
)
