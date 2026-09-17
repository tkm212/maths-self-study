"""Filter controls for the tangent distance page."""

from __future__ import annotations

import dl_ch07_helpers as helpers
from dash import html

from maths_self_study.dashboards.components import filter_bar, slider


def build_filters() -> html.Div:
    return filter_bar(
        slider("td-shift", "Horizontal shift", 0.05, 0.8, helpers.TANGENT_SHIFT_DEFAULT, step=0.05),
    )
