"""Triple-barrier labeling dashboard page."""

from __future__ import annotations

from fml_ch3_pages.triple_barrier.callbacks import register_callbacks
from fml_ch3_pages.triple_barrier.filters import build_filters
from maths_self_study.dashboards.page_factory import define_page
from maths_self_study.viz.textbooks.financial_machine_learning.ch3.algorithms import (
    TRIPLE_BARRIER as TRIPLE_BARRIER_ALGORITHM,
)
from maths_self_study.viz.textbooks.financial_machine_learning.ch3.definitions import (
    TRIPLE_BARRIER as TRIPLE_BARRIER_DEFINITIONS,
)
from maths_self_study.viz.textbooks.financial_machine_learning.ch3.observations import (
    TRIPLE_BARRIER as TRIPLE_BARRIER_OBSERVATIONS,
)
from maths_self_study.viz.textbooks.financial_machine_learning.ch3.theorems import (
    TRIPLE_BARRIER as TRIPLE_BARRIER_THEOREMS,
)

TripleBarrierPage = define_page(
    label="Triple barrier",
    value="triple_barrier",
    title="Triple-barrier labels",
    caption="Snippet 3.2 — Path-dependent targets from profit, stop, and time barriers.",
    methodology=[
        r"Barriers can be volatility-scaled (ATR) in production; here they are fixed fractions.",
    ],
    algorithm=TRIPLE_BARRIER_ALGORITHM,
    definitions=TRIPLE_BARRIER_DEFINITIONS,
    theorems=TRIPLE_BARRIER_THEOREMS,
    observations=TRIPLE_BARRIER_OBSERVATIONS,
    build_filters=build_filters,
    register_callbacks=register_callbacks,
)
