"""Bayes' rule dashboard page."""

from __future__ import annotations

from ch3_pages.bayes.callbacks import register_callbacks
from ch3_pages.bayes.filters import build_filters
from maths_self_study.dashboards.page_factory import define_page

BayesPage = define_page(
    label="Bayes' rule",
    value="bayes",
    title="Bayes' rule — invert conditioning",
    caption="§3.11 — Prior x likelihood → posterior. Base rates dominate rare-disease tests.",
    methodology=[
        "Set prior P(disease), test sensitivity, and false-positive rate for the medical-screening demo.",
        "Multiply prior × likelihood pointwise, then renormalise to get the posterior P(disease | +).",
        "Compare posterior to prior — a positive test rarely means disease when the base rate is low.",
        "In Monty Hall, pick a door and which one is opened; read P(win) per door after conditioning.",
    ],
    build_filters=build_filters,
    register_callbacks=register_callbacks,
)
