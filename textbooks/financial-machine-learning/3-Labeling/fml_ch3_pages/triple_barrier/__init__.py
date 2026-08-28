"""Triple-barrier labeling dashboard page."""

from __future__ import annotations

from fml_ch3_pages.triple_barrier.callbacks import register_callbacks
from fml_ch3_pages.triple_barrier.filters import build_filters
from maths_self_study.dashboards.page_factory import define_page
from maths_self_study.viz.textbooks.financial_machine_learning.ch3.definitions import (
    TRIPLE_BARRIER as TRIPLE_BARRIER_DEFINITIONS,
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
        "CUSUM events reduce overlap vs labeling every bar.",
        "Upper barrier: profit take at P₀(1 + pt). Lower: stop at P₀(1 − sl). Vertical: max hold.",
        "First-touch rule assigns +1, −1, or 0.",
        "Barriers can be volatility-scaled (ATR) in production; here they are fixed fractions.",
    ],
    definitions=TRIPLE_BARRIER_DEFINITIONS,
    theorems=TRIPLE_BARRIER_THEOREMS,
    build_filters=build_filters,
    register_callbacks=register_callbacks,
)
