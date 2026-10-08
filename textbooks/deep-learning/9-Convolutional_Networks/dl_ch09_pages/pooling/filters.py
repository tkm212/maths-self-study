"""Filter controls for the pooling page."""

from __future__ import annotations

import dl_ch09_helpers as helpers
from dash import html

from maths_self_study.dashboards.components import dropdown, filter_bar, slider


def build_filters() -> html.Div:
    return filter_bar(
        dropdown(
            "pool-mode",
            "Pooling mode",
            [
                {"label": "max", "value": "max"},
                {"label": "average", "value": "avg"},
            ],
            "max",
        ),
        slider("pool-size", "Window size", 2, 4, helpers.POOL_SIZE_DEFAULT, step=1),
        slider("pool-stride", "Stride", 1, 4, helpers.POOL_STRIDE_DEFAULT, step=1),
    )
