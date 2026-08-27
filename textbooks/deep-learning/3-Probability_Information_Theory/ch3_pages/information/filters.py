"""Filter controls for the information theory page."""

from __future__ import annotations

from dash import html

from maths_self_study.dashboards.components import filter_bar, prob_simplex, section
from maths_self_study.demos.deep_learning import ch3 as helpers


def build_filters() -> html.Div:
    p, q = helpers.INFO_P, helpers.INFO_Q
    labels = ["a", "b", "c", "d"]
    p_items = [(f"P({label})", float(prob)) for label, prob in zip(labels, p, strict=True)]
    q_items = [(f"Q({label})", float(prob)) for label, prob in zip(labels, q, strict=True)]
    return filter_bar(
        section("P (true)", *prob_simplex("info-p", p_items)),
        section("Q (model)", *prob_simplex("info-q", q_items)),
    )
