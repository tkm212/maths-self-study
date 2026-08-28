"""Concurrency dashboard page."""

from __future__ import annotations

from fml_ch4_pages.concurrency.callbacks import register_callbacks
from fml_ch4_pages.concurrency.filters import build_filters
from maths_self_study.dashboards.page_factory import define_page
from maths_self_study.viz.textbooks.financial_machine_learning.ch4.definitions import (
    CONCURRENCY as CONCURRENCY_DEFINITIONS,
)
from maths_self_study.viz.textbooks.financial_machine_learning.ch4.theorems import (
    CONCURRENCY as CONCURRENCY_THEOREMS,
)

ConcurrencyPage = define_page(
    label="Concurrency",
    value="concurrency",
    title="Label concurrency",
    caption="Ch. 4 — Overlapping triple-barrier events break the IID assumption.",
    methodology=[
        "For each bar, count active labels c(t) between event start and exit.",
        "Peaks in c(t) mark periods where many events share information.",
        "Average uniqueness uses 1/c(t) over each event's lifetime.",
        "Visualize concurrency alongside price to see crowded regimes.",
    ],
    definitions=CONCURRENCY_DEFINITIONS,
    theorems=CONCURRENCY_THEOREMS,
    build_filters=build_filters,
    register_callbacks=register_callbacks,
)
