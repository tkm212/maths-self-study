"""Structured models dashboard page."""

from __future__ import annotations

from ch3_pages.markov.callbacks import register_callbacks
from ch3_pages.markov.filters import build_filters
from maths_self_study.dashboards.page_factory import define_page

MarkovPage = define_page(
    label="Structured models",
    value="markov",
    title="Structured models — factor the joint",
    caption="§3.14 — Each edge is a conditional. RNNs / HMMs / autoregressive LMs are this with neural conditionals.",
    build_filters=build_filters,
    register_callbacks=register_callbacks,
)
