"""Logistic regression dashboard page."""

from __future__ import annotations

from ch4_pages.logistic_regression.callbacks import register_callbacks
from ch4_pages.logistic_regression.filters import build_filters
from maths_self_study.dashboards.page_factory import define_page

LogisticRegressionPage = define_page(
    label="Logistic regression",
    value="logistic_regression",
    title="L1 / L2 regularisation paths",
    caption="§4.4 — Logistic regression on TMDB revenue class.",
    methodology=[
        "Log-odds modelled as linear in x; IRLS maximises concave log-likelihood.",
        "L1 zeros coefficients; L2 shrinks all features toward zero.",
    ],
    build_filters=build_filters,
    register_callbacks=register_callbacks,
)
