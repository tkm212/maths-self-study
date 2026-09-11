"""Stochastic gradient descent dashboard page."""

from __future__ import annotations

from ch5_pages.sgd.callbacks import register_callbacks
from ch5_pages.sgd.filters import build_filters
from maths_self_study.dashboards.page_factory import define_page
from maths_self_study.viz.textbooks.deep_learning.ch5.algorithms import SGD as SGD_ALGORITHM
from maths_self_study.viz.textbooks.deep_learning.ch5.definitions import SGD as SGD_DEFINITIONS

SgdPage = define_page(
    label="SGD",
    value="sgd",
    title="Stochastic gradient descent",
    caption="§5.9 — Each step averages the loss over a random mini-batch B instead of the full dataset.",
    summary=(
        "Stochastic gradient descent approximates the full gradient using a "
        "random mini-batch each step, making large-scale training feasible. "
        "Noise in the gradient can even help escape poor local minima. We use "
        "it as the standard optimiser for neural networks and large "
        "datasets."
    ),
    algorithm=SGD_ALGORITHM,
    definitions=SGD_DEFINITIONS,
    build_filters=build_filters,
    register_callbacks=register_callbacks,
)
