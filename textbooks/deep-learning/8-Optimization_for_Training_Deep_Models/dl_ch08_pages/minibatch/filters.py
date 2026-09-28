"""Filter controls for the mini-batch page."""

from __future__ import annotations

import dl_ch08_helpers as helpers
from dash import html

from maths_self_study.dashboards.components import filter_bar, slider


def build_filters() -> html.Div:
    return filter_bar(
        slider(
            "mb-size",
            "Mini-batch size m",
            1,
            40,
            helpers.BATCH_SIZE_DEFAULT,
            step=1,
        ),
    )
