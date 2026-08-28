"""CUSUM filter dashboard page."""

from __future__ import annotations

from fml_ch2_pages.cusum.callbacks import register_callbacks
from fml_ch2_pages.cusum.filters import build_filters
from maths_self_study.dashboards.page_factory import define_page
from maths_self_study.viz.textbooks.financial_machine_learning.ch2.definitions import CUSUM as CUSUM_DEFINITIONS
from maths_self_study.viz.textbooks.financial_machine_learning.ch2.theorems import CUSUM as CUSUM_THEOREMS

CusumPage = define_page(
    label="CUSUM",
    value="cusum",
    title="CUSUM event filter",
    caption="Snippet 2.4 — Detect cumulative log-return divergences without redundant triggers.",
    methodology=[
        "Track cumulative signed log-return S_t from a reset level of zero.",
        "Sample bar t when |S_t| ≥ h; reset S_t to 0 after each event.",
        "Events mark meaningful shifts — use as labeling seeds instead of every bar.",
        "Threshold h trades sensitivity vs number of events.",
    ],
    definitions=CUSUM_DEFINITIONS,
    theorems=CUSUM_THEOREMS,
    build_filters=build_filters,
    register_callbacks=register_callbacks,
)
