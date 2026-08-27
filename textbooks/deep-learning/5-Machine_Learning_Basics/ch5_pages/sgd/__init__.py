"""Stochastic gradient descent dashboard page."""

from __future__ import annotations

from ch5_pages.sgd.callbacks import register_callbacks
from ch5_pages.sgd.filters import build_filters
from maths_self_study.dashboards.page_factory import define_page

SgdPage = define_page(
    label="SGD",
    value="sgd",
    title="Stochastic gradient descent",
    caption="§5.9 — Mini-batch updates trade off noisy steps against per-epoch cost.",
    methodology=[
        "Full-batch gradient descent uses the entire dataset each step — accurate but expensive.",
        "SGD uses a random mini-batch; gradients are noisy but cheap per iteration.",
        "Learning rate eta controls step size; batch size controls gradient variance.",
    ],
    build_filters=build_filters,
    register_callbacks=register_callbacks,
)
