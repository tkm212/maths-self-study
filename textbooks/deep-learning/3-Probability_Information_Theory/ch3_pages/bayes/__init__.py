"""Bayes' rule dashboard page."""

from __future__ import annotations

from ch3_pages.bayes.callbacks import register_callbacks
from ch3_pages.bayes.filters import build_filters
from maths_self_study.dashboards.page_factory import define_page
from maths_self_study.viz.definitions.ch3 import BAYES as BAYES_DEFINITIONS
from maths_self_study.viz.theorems.ch3 import BAYES as BAYES_THEOREMS

BayesPage = define_page(
    label="Bayes' rule",
    value="bayes",
    title="Bayes' rule — invert conditioning",
    caption="§3.11 — Prior x likelihood → posterior. Base rates dominate rare-disease tests.",
    methodology=[
        "Bayes' rule: P(H | E) = P(E | H) P(H) / P(E). The denominator P(E) = Σ_H P(E | H) P(H) normalises.",
        "Posterior ∝ prior × likelihood — update beliefs about H after observing evidence E.",
        "Base rate P(H) matters: even a sensitive test yields low P(H | +) when the disease is rare.",
        "Compare posterior P(disease | +) to the prior — evidence shifts beliefs, but rarely overturns a low base rate.",
    ],
    definitions=BAYES_DEFINITIONS,
    theorems=BAYES_THEOREMS,
    build_filters=build_filters,
    register_callbacks=register_callbacks,
)
