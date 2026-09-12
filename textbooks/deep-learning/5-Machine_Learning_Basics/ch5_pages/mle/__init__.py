"""Maximum likelihood estimation dashboard page."""

from __future__ import annotations

from ch5_pages.mle.callbacks import register_callbacks
from ch5_pages.mle.filters import build_filters
from maths_self_study.dashboards.page_factory import define_page
from maths_self_study.viz.textbooks.deep_learning.ch5.definitions import MLE as MLE_DEFINITIONS
from maths_self_study.viz.textbooks.deep_learning.ch5.proofs import MLE as MLE_PROOF
from maths_self_study.viz.textbooks.deep_learning.ch5.theorems import MLE as MLE_THEOREMS

MlePage = define_page(
    label="MLE",
    value="mle",
    title="Maximum likelihood estimation",
    caption="§5.5 — MLE picks parameters that make the observed data most probable.",
    summary=(
        "Maximum likelihood estimation chooses parameters that make the "
        "observed data most probable under a specified model. It connects "
        "probability theory directly to parameter fitting. We use MLE as the "
        "default principle for training many statistical and deep learning "
        "models."
    ),
    methodology=[
        "Slide to shift every sample and watch the MLE mean and fitted bell curve move together.",
        "MLE is consistent — with enough data, estimates converge to true parameters.",
    ],
    definitions=MLE_DEFINITIONS,
    theorems=MLE_THEOREMS,
    proof=MLE_PROOF,
    build_filters=build_filters,
    register_callbacks=register_callbacks,
)
