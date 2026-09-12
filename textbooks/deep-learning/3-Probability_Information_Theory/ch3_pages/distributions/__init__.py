"""Distributions dashboard page."""

from __future__ import annotations

from ch3_pages.distributions.callbacks import register_callbacks
from ch3_pages.distributions.filters import build_filters
from maths_self_study.dashboards.page_factory import define_page
from maths_self_study.viz.textbooks.deep_learning.ch3.definitions import DISTRIBUTIONS as DISTRIBUTIONS_DEFINITIONS

DistributionsPage = define_page(
    label="Distributions",
    value="dist",
    title="The distributions deep learning lives on",
    caption="§3.9 — Bernoulli (one bit), Categorical (k classes), Gaussian (continuous workhorse).",
    summary=(
        "A handful of distributions - Bernoulli, categorical, Gaussian - "
        "cover most of what deep learning builds on. They specify how "
        "outputs, labels, and noise are generated. We use them as the "
        "building blocks for loss functions and generative models."
    ),
    methodology=[
        "Adjust categorical probabilities and the bivariate covariance matrix to see entropy, PMF, and contour geometry change.",
    ],
    definitions=DISTRIBUTIONS_DEFINITIONS,
    build_filters=build_filters,
    register_callbacks=register_callbacks,
)
