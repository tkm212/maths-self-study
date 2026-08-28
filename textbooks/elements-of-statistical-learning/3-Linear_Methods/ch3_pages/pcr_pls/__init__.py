"""PCR / PLS dashboard page."""

from __future__ import annotations

from ch3_pages.pcr_pls.callbacks import register_callbacks
from ch3_pages.pcr_pls.filters import build_filters
from maths_self_study.dashboards.page_factory import define_page

PcrPlsPage = define_page(
    label="PCR / PLS",
    value="pcr_pls",
    title="Dimension reduction regression",
    caption="§3.5 — PCR and PLS on TMDB revenue.",
    methodology=[
        "Both project p features into M ≪ p directions then fit OLS.",
        "Beyond optimal M, test MSE rises from noise fitting.",
    ],
    build_filters=build_filters,
    register_callbacks=register_callbacks,
)
