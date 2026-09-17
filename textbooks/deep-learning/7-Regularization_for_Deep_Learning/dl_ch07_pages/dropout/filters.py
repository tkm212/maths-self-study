"""Filter controls for the dropout page."""

from __future__ import annotations

import dl_ch07_helpers as helpers
from dash import html

from maths_self_study.dashboards.components import filter_bar, slider


def build_filters() -> html.Div:
    return filter_bar(
        slider("do-rate", "Dropout rate p", 0.0, 0.75, helpers.DROPOUT_DEFAULT, step=0.05),
    )
