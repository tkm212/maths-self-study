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
    build_filters=build_filters,
    register_callbacks=register_callbacks,
)
