"""Filter controls for the translation page."""

from __future__ import annotations

import dl_ch09_helpers as helpers
from dash import html

from maths_self_study.dashboards.components import filter_bar, slider


def build_filters() -> html.Div:
    return filter_bar(
        slider("trans-shift", "Shift (pixels)", 0, 4, helpers.SHIFT_DEFAULT, step=1),
        slider("trans-padding", "Zero-padding P", 0, 3, 1, step=1),
    )
