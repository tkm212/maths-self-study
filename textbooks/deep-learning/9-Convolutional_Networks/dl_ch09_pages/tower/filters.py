"""Filter controls for the tower page."""

from __future__ import annotations

import dl_ch09_helpers as helpers
from dash import html

from maths_self_study.dashboards.components import filter_bar, slider


def build_filters() -> html.Div:
    return filter_bar(
        slider("tower-blocks", "Conv+pool blocks", 1, 6, helpers.TOWER_BLOCKS_DEFAULT, step=1),
        slider("tower-kernel", "Kernel size", 3, 5, helpers.KERNEL_SIZE_DEFAULT, step=2),
    )
