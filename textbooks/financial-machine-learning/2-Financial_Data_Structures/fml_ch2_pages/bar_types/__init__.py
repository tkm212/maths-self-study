"""Bar types dashboard page."""

from __future__ import annotations

from fml_ch2_pages.bar_types.callbacks import register_callbacks
from fml_ch2_pages.bar_types.filters import build_filters
from maths_self_study.dashboards.page_factory import define_page
from maths_self_study.viz.textbooks.financial_machine_learning.ch2.definitions import BAR_TYPES as BAR_DEFINITIONS
from maths_self_study.viz.textbooks.financial_machine_learning.ch2.theorems import BAR_TYPES as BAR_THEOREMS

BarTypesPage = define_page(
    label="Bar types",
    value="bar_types",
    title="Information-driven bars",
    caption="Ch. 2 — Compare time, tick, volume, and dollar bars on BTC tick data.",
    methodology=[
        "Time bars sample at fixed clock intervals — familiar but activity-varying.",
        "Tick bars normalize by transaction count; volume and dollar bars normalize by size and information flow.",
        "Mandelbrot–Taylor: transaction-based sampling yields returns closer to Gaussian IID.",
        "Dollar bars are the book's preferred information bars for downstream labeling.",
    ],
    definitions=BAR_DEFINITIONS,
    theorems=BAR_THEOREMS,
    build_filters=build_filters,
    register_callbacks=register_callbacks,
)
