"""Kernel density dashboard page."""

from __future__ import annotations

from ch6_pages.kernel_density.callbacks import register_callbacks
from ch6_pages.kernel_density.filters import build_filters
from maths_self_study.dashboards.page_factory import define_page
from maths_self_study.viz.textbooks.elements_of_statistical_learning.ch6.definitions import (
    KERNEL_DENSITY as KERNEL_DENSITY_DEFINITIONS,
)

KernelDensityPage = define_page(
    label="Kernel density",
    value="kernel_density",
    title="KDE and Naive Bayes",
    caption="§6.6 — Density estimation and generative classification.",
    summary=(
        "Kernel density estimation smooths out a histogram to estimate where probability "
        "mass concentrates in the data. Each observation contributes a small bump; "
        "the sum gives a continuous density curve. We use it to explore distributions "
        "nonparametrically and as the building block for generative classifiers."
    ),
    methodology=[
        "KDE in log₁p-space with back-transformed x-axis.",
        "Posterior ∝ πₖ f̂ₖ(x) from class-conditional densities.",
    ],
    definitions=KERNEL_DENSITY_DEFINITIONS,
    build_filters=build_filters,
    register_callbacks=register_callbacks,
)
