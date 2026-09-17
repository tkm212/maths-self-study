"""Mini-batch SGD page."""

from __future__ import annotations

from dl_ch08_pages.minibatch.callbacks import register_callbacks
from dl_ch08_pages.minibatch.filters import build_filters
from maths_self_study.dashboards.page_factory import define_page
from maths_self_study.viz.textbooks.deep_learning.ch8.algorithms import MINIBATCH_SGD as MINIBATCH_ALGORITHM
from maths_self_study.viz.textbooks.deep_learning.ch8.definitions import MINIBATCH as MINIBATCH_DEFINITIONS

MinibatchPage = define_page(
    label="Mini-batch",
    value="minibatch",
    title="Mini-batch gradient descent",
    caption="§8.1.3 — Batch size trades gradient noise against compute per step.",
    summary=(
        "Stochastic methods estimate the gradient from a subset of examples. "
        "Small batches add noise and cheap updates; full-batch steps are smooth "
        "but expensive. Both paths are compared on linear regression."
    ),
    methodology=[
        "Fit a scalar linear model with mean-squared error.",
        "Plot full-data MSE after each SGD update for two batch sizes.",
        "Keep learning rate and step count fixed for a fair comparison.",
    ],
    definitions=MINIBATCH_DEFINITIONS,
    algorithm=MINIBATCH_ALGORITHM,
    build_filters=build_filters,
    register_callbacks=register_callbacks,
)
