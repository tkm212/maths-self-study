"""Markov chains dashboard page."""

from __future__ import annotations

from ch3_pages.markov.callbacks import register_callbacks
from ch3_pages.markov.filters import build_filters
from maths_self_study.dashboards.page_factory import define_page
from maths_self_study.viz.textbooks.deep_learning.ch3.definitions import MARKOV as MARKOV_DEFINITIONS
from maths_self_study.viz.textbooks.deep_learning.ch3.theorems import MARKOV as MARKOV_THEOREMS

MarkovPage = define_page(
    label="Markov chains",
    value="markov",
    title="Structured probability — factorisation",
    caption="§3.10 — Chain rule factorises joints; Markov property drops distant history.",
    summary=(
        "Markov chains model sequences where each state depends only on the "
        "previous one. The chain rule factorises any joint distribution into "
        "conditional pieces. We use this structure in language models, "
        "time-series, and autoregressive generation."
    ),
    methodology=[
        "Adjust transition probabilities and inspect how the joint factorises along the chain graph.",
        "Each edge is one conditional factor; RNNs and autoregressive models use the same pattern with neural nets as conditionals.",
    ],
    definitions=MARKOV_DEFINITIONS,
    theorems=MARKOV_THEOREMS,
    build_filters=build_filters,
    register_callbacks=register_callbacks,
)
