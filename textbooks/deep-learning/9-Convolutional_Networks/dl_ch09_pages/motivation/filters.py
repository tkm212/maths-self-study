"""Filter controls for the motivation page."""

from __future__ import annotations

import dl_ch09_helpers as helpers
from dash import html

from maths_self_study.dashboards.components import dropdown, filter_bar, slider
from maths_self_study.math.convolutional import KERNEL_PRESETS

_FILTER_OPTS = [{"label": k.replace("_", " "), "value": k} for k in KERNEL_PRESETS]


def build_filters() -> html.Div:
    return filter_bar(
        dropdown("mot-filter", "Edge kernel", _FILTER_OPTS, helpers.FILTER_DEFAULT),
        slider("mot-shift", "Translation shift (px)", 0, 4, helpers.SHIFT_DEFAULT, step=1),
        slider("mot-blocks", "Conv+pool blocks (RF)", 1, 5, helpers.TOWER_BLOCKS_DEFAULT, step=1),
    )
