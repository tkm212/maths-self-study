"""Kernel density dashboard page."""

from __future__ import annotations

from ch6_pages.kernel_density.callbacks import register_callbacks
from ch6_pages.kernel_density.filters import build_filters
from maths_self_study.dashboards.page_factory import define_page

KernelDensityPage = define_page(
    label="Kernel density",
    value="kernel_density",
    title="KDE and Naive Bayes",
    caption="§6.6 — Density estimation and generative classification.",
    methodology=[
        "KDE in log₁p-space with back-transformed x-axis.",
        "Posterior ∝ πₖ f̂ₖ(x) from class-conditional densities.",
    ],
    build_filters=build_filters,
    register_callbacks=register_callbacks,
)
