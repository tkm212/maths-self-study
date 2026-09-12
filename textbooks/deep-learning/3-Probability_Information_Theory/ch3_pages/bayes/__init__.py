"""Bayes' rule dashboard page."""

from __future__ import annotations

from ch3_pages.bayes.callbacks import register_callbacks
from ch3_pages.bayes.filters import build_filters
from maths_self_study.dashboards.page_factory import define_page
from maths_self_study.viz.textbooks.deep_learning.ch3.definitions import BAYES as BAYES_DEFINITIONS
from maths_self_study.viz.textbooks.deep_learning.ch3.proofs import BAYES as BAYES_PROOF
from maths_self_study.viz.textbooks.deep_learning.ch3.theorems import BAYES as BAYES_THEOREMS

BayesPage = define_page(
    label="Bayes' rule",
    value="bayes",
    title="Bayes' rule — invert conditioning",
    caption="§3.11 — Prior x likelihood → posterior. Base rates dominate rare-disease tests.",
    summary=(
        "Bayes' rule inverts conditional probability - updating beliefs about "
        "causes given observed evidence. The prior, likelihood, and posterior "
        "combine prior knowledge with data. We use it whenever we need "
        "principled uncertainty quantification rather than point estimates "
        "alone."
    ),
    methodology=[
        "Slide prior disease rate, sensitivity, and false-positive rate — compare posterior to prior after a positive test.",
        "Base rates matter: even a sensitive test yields low posterior probability when the disease is rare.",
    ],
    definitions=BAYES_DEFINITIONS,
    theorems=BAYES_THEOREMS,
    proof=BAYES_PROOF,
    build_filters=build_filters,
    register_callbacks=register_callbacks,
)
