"""Distributions dashboard page."""

from __future__ import annotations

from ch3_pages.distributions.callbacks import register_callbacks
from ch3_pages.distributions.filters import build_filters
from maths_self_study.dashboards.page_factory import define_page

DistributionsPage = define_page(
    label="Distributions",
    value="dist",
    title="The distributions deep learning lives on",
    caption="§3.9 — Bernoulli (one bit), Categorical (k classes), Gaussian (continuous workhorse).",
    methodology=[
        "Inspect binary entropy H(p) — uncertainty peaks at p = ½ (Bernoulli / sigmoid outputs).",
        "Compare 1D and 2D Gaussian PDFs; elliptical contours come from the covariance matrix.",
        "Edit the 2×2 covariance and read its eigenvalues — axis lengths of the density ellipses.",
        "Set categorical probabilities (softmax targets) and confirm they sum to 1 over finite support.",
    ],
    build_filters=build_filters,
    register_callbacks=register_callbacks,
)
