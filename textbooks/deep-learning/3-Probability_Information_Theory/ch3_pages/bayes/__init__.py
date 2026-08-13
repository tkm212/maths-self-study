"""Bayes' rule dashboard page."""

from __future__ import annotations

from ch3_pages._page_factory import page
from ch3_pages.bayes.callbacks import register_callbacks
from ch3_pages.bayes.filters import build_filters

BayesPage = page(
    label="Bayes' rule",
    value="bayes",
    title="Bayes' rule — invert conditioning",
    caption="§3.11 — Prior x likelihood → posterior. Base rates dominate rare-disease tests.",
    build_filters=build_filters,
    register_callbacks=register_callbacks,
)
