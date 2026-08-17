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
    methodology=[
        "Set P(X₁) and transition tables P(X₂ | X₁), P(X₃ | X₂) for a three-node binary chain.",
        "Multiply local conditionals along the chain to build the full joint P(X₁, X₂, X₃).",
        "Inspect the chain diagram — each edge is one conditional factor in the joint factorisation.",
        "Confirm joint entries sum to 1 and read marginal queries such as P(X₃ = 1).",
    ],
    build_filters=build_filters,
    register_callbacks=register_callbacks,
)
