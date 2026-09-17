"""Filter controls for the initialization page."""

from __future__ import annotations

import dl_ch08_helpers as helpers
from dash import html

from maths_self_study.dashboards.components import filter_bar, slider


def build_filters() -> html.Div:
    return filter_bar(
        slider(
            "init-scale",
            "Init scale (× Xavier std)",
            0.05,
            4.0,
            helpers.INIT_SCALE_DEFAULT,
            step=0.05,
        ),
    )
