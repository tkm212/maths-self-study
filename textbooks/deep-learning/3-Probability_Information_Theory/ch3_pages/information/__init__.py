"""Information theory dashboard page."""

from __future__ import annotations

from ch3_pages.information.callbacks import register_callbacks
from ch3_pages.information.filters import build_filters
from maths_self_study.dashboards.page_factory import define_page
from maths_self_study.viz.textbooks.deep_learning.ch3.definitions import INFORMATION as INFORMATION_DEFINITIONS
from maths_self_study.viz.textbooks.deep_learning.ch3.proofs import GIBBS as GIBBS_PROOF
from maths_self_study.viz.textbooks.deep_learning.ch3.theorems import INFORMATION as INFORMATION_THEOREMS

InformationPage = define_page(
    label="Information theory",
    value="info",
    title="Information and surprise",
    caption="§3.13 — I(x) = -log P(x). H(P) averages surprise; H(P,Q) is classification loss; KL is asymmetric.",
    summary=(
        "Information theory quantifies surprise: rare events carry more "
        "information than expected ones. Entropy averages surprise; cross-"
        "entropy measures prediction quality; KL divergence compares "
        "distributions. We use these quantities as natural training "
        "objectives and diagnostics."
    ),
    methodology=[
        "Adjust the probability tables and compare self-information, entropy, cross-entropy, and KL in the summary table.",
        "KL is asymmetric — swapping P and Q changes the value.",
    ],
    definitions=INFORMATION_DEFINITIONS,
    theorems=INFORMATION_THEOREMS,
    proof=GIBBS_PROOF,
    build_filters=build_filters,
    register_callbacks=register_callbacks,
)
