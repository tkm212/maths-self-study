"""Filter controls for the MLE page."""

from __future__ import annotations

from dash import html

from maths_self_study.dashboards.components import filter_bar, slider


def build_filters() -> html.Div:
    return filter_bar(
        slider("mle-shift", "Shift all samples by", -1.0, 1.0, 0.0, step=0.05),
    )
