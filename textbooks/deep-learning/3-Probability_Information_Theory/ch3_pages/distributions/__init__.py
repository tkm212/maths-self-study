"""Distributions dashboard page."""

from __future__ import annotations

from ch3_pages.distributions.callbacks import register_callbacks
from ch3_pages.distributions.filters import build_filters
from maths_self_study.dashboards.page_factory import define_page

DistributionsPage = define_page(
    label="Distributions",
    value="dist",
    title="The distributions deep learning lives on",
    caption="§3.9 — Bernoulli (one bit), Categorical (k classes), Gaussian (continuous workhorse).",
    build_filters=build_filters,
    register_callbacks=register_callbacks,
)
