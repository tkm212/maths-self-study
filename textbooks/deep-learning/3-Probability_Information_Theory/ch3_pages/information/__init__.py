"""Information theory dashboard page."""

from __future__ import annotations

from ch3_pages.information.callbacks import register_callbacks
from ch3_pages.information.filters import build_filters
from maths_self_study.dashboards.page_factory import define_page

InformationPage = define_page(
    label="Information theory",
    value="info",
    title="Information and surprise",
    caption="§3.13 — I(x) = -log P(x). H(P) averages surprise; H(P,Q) is classification loss; KL is asymmetric.",
    methodology=[
        "Set true distribution P and read self-information −log P(x) — rare events carry more surprise.",
        "Compare model Q to P; cross-entropy H(P, Q) is the expected code length under Q for data from P.",
        "KL(P ‖ Q) measures extra surprise from using Q instead of P — asymmetric, not a metric.",
        "Use the bar chart to see which outcomes contribute most to H(P, Q) and KL(P ‖ Q).",
    ],
    build_filters=build_filters,
    register_callbacks=register_callbacks,
)
