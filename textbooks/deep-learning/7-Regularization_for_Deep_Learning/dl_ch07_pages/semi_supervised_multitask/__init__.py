"""Semi-supervised and multitask learning page."""

from __future__ import annotations

from dl_ch07_pages.semi_supervised_multitask.callbacks import register_callbacks
from dl_ch07_pages.semi_supervised_multitask.filters import build_filters
from maths_self_study.dashboards.page_factory import define_page
from maths_self_study.viz.textbooks.deep_learning.ch7.definitions import (
    SEMI_SUPERVISED_MULTITASK as SEMI_SUPERVISED_MULTITASK_DEFINITIONS,
)

SemiSupervisedMultitaskPage = define_page(
    label="Semi-supervised / multitask",
    value="semi_supervised_multitask",
    title="Semi-supervised and multitask learning",
    caption="§7.6–7.7 — Use unlabeled data and shared tasks to improve generalization.",
    summary=(
        "Semi-supervised learning exploits unlabeled examples with an auxiliary "
        "objective; here, predictions should be stable under input noise. Multitask "
        "learning trains related tasks with a shared representation so each task "
        "benefits from inductive transfer."
    ),
    methodology=[
        "Train on a subset of labeled points only as a baseline.",
        "Add consistency regularization on unlabeled inputs.",
        "Compare shared vs separate hidden layers on two related regression tasks.",
    ],
    definitions=SEMI_SUPERVISED_MULTITASK_DEFINITIONS,
    build_filters=build_filters,
    register_callbacks=register_callbacks,
)
