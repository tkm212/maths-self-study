"""Universal approximation page."""

from __future__ import annotations

from dl_ch06_pages.universal_approx.callbacks import register_callbacks
from dl_ch06_pages.universal_approx.filters import build_filters
from maths_self_study.dashboards.page_factory import define_page
from maths_self_study.viz.textbooks.deep_learning.ch6.definitions import UNIVERSAL_APPROX as UNIVERSAL_DEFINITIONS

UniversalApproxPage = define_page(
    label="Universal approx",
    value="universal_approx",
    title="Universal approximation",
    caption="§6.4.1 — A wide hidden layer can approximate smooth functions.",
    summary=(
        "The universal approximation theorem guarantees that a feedforward "
        "network with one sufficiently wide hidden layer can approximate any "
        "continuous function on a compact set. In practice, width controls "
        "how closely a shallow MLP fits complex targets — more hidden units "
        "generally reduce training error on smooth curves."
    ),
    overview_parts=[
        (
            "Cybenko-Hornik",
            "Wide single-hidden-layer nets with nonpolynomial activation approximate continuous functions on compact sets.",
        ),
        (
            "Width vs depth",
            "Theory needs many hidden units; practice often prefers deeper narrow nets for the same accuracy.",
        ),
        (
            "Empirical fit",
            "Training error falls as hidden width grows on smooth targets in the demo.",
        ),
    ],
    methodology=[
        "One hidden layer + nonlinear activation is a universal approximator (Cybenko, Hornik).",
        "Increasing hidden width adds basis functions the network can combine.",
        "Depth can achieve similar expressivity with fewer parameters on structured tasks.",
    ],
    definitions=UNIVERSAL_DEFINITIONS,
    build_filters=build_filters,
    register_callbacks=register_callbacks,
)
