"""Maximum likelihood estimation dashboard page."""

from __future__ import annotations

from ch5_pages.mle.callbacks import register_callbacks
from ch5_pages.mle.filters import build_filters
from maths_self_study.dashboards.page_factory import define_page
from maths_self_study.viz.definitions.ch5 import MLE as MLE_DEFINITIONS

MlePage = define_page(
    label="MLE",
    value="mle",
    title="Maximum likelihood estimation",
    caption="§5.5 — MLE picks parameters that make the observed data most probable.",
    methodology=[
        "For a Gaussian, MLE sets mu to the sample mean and sigma^2 to the average squared deviation.",
        "Likelihood is the probability of the data given parameters; we maximize it over mu and sigma.",
        "MLE is consistent — with enough data, estimates converge to true parameters.",
    ],
    definitions=MLE_DEFINITIONS,
    build_filters=build_filters,
    register_callbacks=register_callbacks,
)
