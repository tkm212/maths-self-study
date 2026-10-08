"""Filter controls for the receptive field page."""

from __future__ import annotations

import dl_ch09_helpers as helpers
from dash import html

from maths_self_study.dashboards.components import filter_bar, slider


def build_filters() -> html.Div:
    return filter_bar(
        slider("rf-blocks", "Conv+pool blocks", 1, 5, helpers.TOWER_BLOCKS_DEFAULT, step=1),
        slider("rf-kernel", "Kernel size", 3, 5, helpers.KERNEL_SIZE_DEFAULT, step=2),
    )
