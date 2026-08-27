"""Structured models dashboard page."""

from __future__ import annotations

from ch3_pages.markov.callbacks import register_callbacks
from ch3_pages.markov.filters import build_filters
from maths_self_study.dashboards.page_factory import define_page
from maths_self_study.viz.definitions.ch3 import MARKOV as MARKOV_DEFINITIONS
from maths_self_study.viz.theorems.ch3 import MARKOV as MARKOV_THEOREMS

MarkovPage = define_page(
    label="Structured models",
    value="markov",
    title="Structured models — factor the joint",
    caption="§3.14 — Each edge is a conditional. RNNs / HMMs / autoregressive LMs are this with neural conditionals.",
    methodology=[
        "A joint over many variables factorises: P(x₁, …, xₙ) = P(x₁) Πᵢ P(xᵢ | x₁, …, xᵢ₋₁) (chain rule).",
        "Markov chain: P(xᵢ | x₁, …, xᵢ₋₁) = P(xᵢ | xᵢ₋₁) — only the previous state matters.",
        "Each edge in the graph is one conditional factor; the product builds the full joint.",
        "RNNs and autoregressive language models use the same factorisation with neural nets as conditionals.",
    ],
    definitions=MARKOV_DEFINITIONS,
    theorems=MARKOV_THEOREMS,
    build_filters=build_filters,
    register_callbacks=register_callbacks,
)
