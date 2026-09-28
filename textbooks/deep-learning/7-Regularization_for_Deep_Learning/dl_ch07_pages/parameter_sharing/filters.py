"""Filter controls for the parameter sharing page."""

from __future__ import annotations

import dl_ch07_helpers as helpers
from dash import html

from maths_self_study.dashboards.components import filter_bar, slider


def build_filters() -> html.Div:
    return filter_bar(
        slider("ps-kernel", "Conv kernel size", 2, 5, helpers.KERNEL_SIZE_DEFAULT, step=1),
    )
