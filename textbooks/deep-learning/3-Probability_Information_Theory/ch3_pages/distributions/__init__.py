"""Distributions dashboard page."""

from __future__ import annotations

from ch3_pages.distributions.callbacks import register_callbacks
from ch3_pages.distributions.filters import build_filters
from maths_self_study.dashboards.page_factory import define_page
from maths_self_study.viz.definitions.ch3 import DISTRIBUTIONS as DISTRIBUTIONS_DEFINITIONS
from maths_self_study.viz.theorems.ch3 import DISTRIBUTIONS as DISTRIBUTIONS_THEOREMS

DistributionsPage = define_page(
    label="Distributions",
    value="dist",
    title="The distributions deep learning lives on",
    caption="§3.9 — Bernoulli (one bit), Categorical (k classes), Gaussian (continuous workhorse).",
    methodology=[
        "Bernoulli: P(X = 1) = p, one binary outcome. Entropy H(p) = −p log p − (1−p) log(1−p) peaks at p = ½.",
        "Categorical: P(X = k) = pₖ with Σ pₖ = 1 over K classes — the target of a softmax output layer.",
        "Gaussian (normal): N(μ, σ²) has PDF ∝ exp(−(x−μ)²/(2σ²)); bivariate N(μ, Σ) has elliptical level sets.",
        "Multivariate covariance Σ sets axis orientation and scale; eigenvalues of Σ are variance along principal axes.",
    ],
    definitions=DISTRIBUTIONS_DEFINITIONS,
    theorems=DISTRIBUTIONS_THEOREMS,
    build_filters=build_filters,
    register_callbacks=register_callbacks,
)
