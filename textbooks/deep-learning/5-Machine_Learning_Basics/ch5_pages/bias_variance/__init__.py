"""Bias-variance tradeoff dashboard page."""

from __future__ import annotations

from ch5_pages.bias_variance.callbacks import register_callbacks
from ch5_pages.bias_variance.filters import build_filters
from maths_self_study.dashboards.page_factory import define_page
from maths_self_study.viz.definitions.ch5 import BIAS_VARIANCE as BIAS_VARIANCE_DEFINITIONS
from maths_self_study.viz.theorems.ch5 import BIAS_VARIANCE as BIAS_VARIANCE_THEOREMS

BiasVariancePage = define_page(
    label="Bias-variance",
    value="bias_variance",
    title="Bias-variance decomposition",
    caption="§5.4 — Test error typically U-shaped in model complexity.",
    methodology=[
        "Bias: error from overly rigid models that miss structure in the data.",
        "Variance: error from fitting noise when the model is too flexible.",
        "Generalization error balances both — the sweet spot minimizes test MSE.",
    ],
    definitions=BIAS_VARIANCE_DEFINITIONS,
    theorems=BIAS_VARIANCE_THEOREMS,
    build_filters=build_filters,
    register_callbacks=register_callbacks,
)
