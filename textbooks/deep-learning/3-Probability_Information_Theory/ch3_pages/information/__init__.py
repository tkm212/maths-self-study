"""Information theory dashboard page."""

from __future__ import annotations

from ch3_pages.information.callbacks import register_callbacks
from ch3_pages.information.filters import build_filters
from maths_self_study.dashboards.page_factory import define_page
from maths_self_study.viz.definitions.ch3 import INFORMATION as INFORMATION_DEFINITIONS

InformationPage = define_page(
    label="Information theory",
    value="info",
    title="Information and surprise",
    caption="§3.13 — I(x) = -log P(x). H(P) averages surprise; H(P,Q) is classification loss; KL is asymmetric.",
    methodology=[
        "Self-information: I(x) = −log P(x) — surprise of outcome x (nats with ln, bits with log₂).",
        "Shannon entropy H(P) = E[−log P(X)] = −Σ P(x) log P(x) — average surprise over P.",
        "Cross-entropy H(P, Q) = E_P[−log Q(X)] — expected code length using Q on P-generated data; softmax + log loss.",
        "KL divergence KL(P ‖ Q) = E_P[log(P/Q)] = H(P, Q) − H(P) ≥ 0, zero iff P = Q; asymmetric, not a metric.",
    ],
    definitions=INFORMATION_DEFINITIONS,
    build_filters=build_filters,
    register_callbacks=register_callbacks,
)
