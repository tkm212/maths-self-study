"""Filter controls for the variants page."""

from __future__ import annotations

import dl_ch09_helpers as helpers
from dash import html

from maths_self_study.dashboards.components import filter_bar, slider


def build_filters() -> html.Div:
    return filter_bar(
        slider("var-dilation", "Dilation d", 1, 4, helpers.DILATION_DEFAULT, step=1),
    )
