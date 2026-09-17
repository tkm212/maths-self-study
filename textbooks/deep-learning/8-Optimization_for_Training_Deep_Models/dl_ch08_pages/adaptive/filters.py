"""Filter controls for the adaptive optimizers page."""

from __future__ import annotations

import dl_ch08_helpers as helpers
from dash import html

from maths_self_study.dashboards.components import filter_bar, slider


def build_filters() -> html.Div:
    return filter_bar(
        slider(
            "adapt-lr",
            "Base learning rate",
            0.005,
            0.08,
            helpers.OPTIMIZER_LR_DEFAULT,
            step=0.005,
        ),
    )
