"""LDA / QDA dashboard page."""

from __future__ import annotations

from ch4_pages.lda.callbacks import register_callbacks
from ch4_pages.lda.filters import build_filters
from maths_self_study.dashboards.page_factory import define_page
from maths_self_study.viz.textbooks.elements_of_statistical_learning.ch4.definitions import (
    LDA as LDA_DEFINITIONS,
)
from maths_self_study.viz.textbooks.elements_of_statistical_learning.ch4.proofs import LDA as LDA_PROOF
from maths_self_study.viz.textbooks.elements_of_statistical_learning.ch4.theorems import (
    LDA as LDA_THEOREMS,
)

LdaPage = define_page(
    label="LDA / QDA",
    value="lda",
    title="Linear and quadratic discriminants",
    caption="§4.3 — Generative classifiers on TMDB revenue class.",
    summary=(
        "Linear discriminant analysis models each class as a bell-shaped cloud of "
        "points and draws the boundary where two class densities are equal. Sharing "
        "one covariance shape across classes gives a straight decision boundary. We "
        "use it when classes are roughly normal and a simple, fast classifier is "
        "enough."
    ),
    methodology=[
        "LDA: shared covariance → linear boundary.",
        "RDA interpolates QDA (α=0) and LDA (α=1).",
    ],
    definitions=LDA_DEFINITIONS,
    theorems=LDA_THEOREMS,
    proof=LDA_PROOF,
    build_filters=build_filters,
    register_callbacks=register_callbacks,
)
