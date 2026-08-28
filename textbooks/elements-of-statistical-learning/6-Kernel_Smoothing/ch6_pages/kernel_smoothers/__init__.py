"""Kernel smoothers dashboard page."""

from __future__ import annotations

from ch6_pages.kernel_smoothers.callbacks import register_callbacks
from ch6_pages.kernel_smoothers.filters import build_filters
from maths_self_study.dashboards.page_factory import define_page
from maths_self_study.viz.textbooks.elements_of_statistical_learning.ch6.definitions import (
    KERNEL_SMOOTHERS as KERNEL_SMOOTHERS_DEFINITIONS,
)
from maths_self_study.viz.textbooks.elements_of_statistical_learning.ch6.theorems import (
    KERNEL_SMOOTHERS as KERNEL_SMOOTHERS_THEOREMS,
)

KernelSmoothersPage = define_page(
    label="Kernel smoothers",
    value="kernel_smoothers",
    title="NW and local polynomials",
    caption="§6.1–6.2 — Kernel smoothers in log₁p space.",
    methodology=[
        "Degree 0 (NW) has boundary bias; local linear corrects it.",
        "LOO-CV exploits the linear smoother hat matrix.",
    ],
    definitions=KERNEL_SMOOTHERS_DEFINITIONS,
    theorems=KERNEL_SMOOTHERS_THEOREMS,
    build_filters=build_filters,
    register_callbacks=register_callbacks,
)
