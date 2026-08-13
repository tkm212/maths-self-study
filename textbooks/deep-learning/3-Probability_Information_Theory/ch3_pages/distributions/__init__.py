"""Distributions dashboard page."""

from __future__ import annotations

from ch3_pages._page_factory import page
from ch3_pages.distributions.callbacks import register_callbacks
from ch3_pages.distributions.filters import build_filters

DistributionsPage = page(
    label="Distributions",
    value="dist",
    title="The distributions deep learning lives on",
    caption="§3.9 — Bernoulli (one bit), Categorical (k classes), Gaussian (continuous workhorse).",
    build_filters=build_filters,
    register_callbacks=register_callbacks,
)
