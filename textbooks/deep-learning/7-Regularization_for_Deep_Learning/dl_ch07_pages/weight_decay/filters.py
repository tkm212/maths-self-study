"""Filter controls for the weight decay page."""

from __future__ import annotations

import dl_ch07_helpers as helpers
from dash import html

from maths_self_study.dashboards.components import filter_bar, slider


def build_filters() -> html.Div:
    return filter_bar(
        slider("wd-l2", "L2 penalty lambda", 0.0, 0.2, helpers.L2_DEFAULT, step=0.005),
    )
