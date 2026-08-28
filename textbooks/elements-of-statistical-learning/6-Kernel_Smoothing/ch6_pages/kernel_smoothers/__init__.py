"""Kernel smoothers dashboard page."""

from __future__ import annotations

from ch6_pages.kernel_smoothers.callbacks import register_callbacks
from ch6_pages.kernel_smoothers.filters import build_filters
from maths_self_study.dashboards.page_factory import define_page

KernelSmoothersPage = define_page(
    label="Kernel smoothers",
    value="kernel_smoothers",
    title="NW and local polynomials",
    caption="§6.1–6.2 — Kernel smoothers in log₁p space.",
    methodology=[
        "Degree 0 (NW) has boundary bias; local linear corrects it.",
        "LOO-CV exploits the linear smoother hat matrix.",
    ],
    build_filters=build_filters,
    register_callbacks=register_callbacks,
)
