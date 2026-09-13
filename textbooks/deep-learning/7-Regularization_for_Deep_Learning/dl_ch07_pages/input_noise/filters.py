"""Filter controls for the input noise page."""

from __future__ import annotations

import dl_ch07_helpers as helpers
from dash import html

from maths_self_study.dashboards.components import filter_bar, slider


def build_filters() -> html.Div:
    return filter_bar(
        slider(
            "in-noise",
            "Input noise sigma",
            0.0,
            0.35,
            helpers.INPUT_NOISE_DEFAULT,
            step=0.02,
        ),
    )
