"""Filter controls for the convolution page."""

from __future__ import annotations

import dl_ch09_helpers as helpers
from dash import html

from maths_self_study.dashboards.components import dropdown, filter_bar, slider
from maths_self_study.math.convolutional import KERNEL_PRESETS

_FILTER_OPTS = [{"label": k.replace("_", " "), "value": k} for k in KERNEL_PRESETS]


def build_filters() -> html.Div:
    return filter_bar(
        dropdown("conv-filter", "Kernel preset", _FILTER_OPTS, helpers.FILTER_DEFAULT),
        slider("conv-stride", "Stride S", 1, 3, helpers.STRIDE_DEFAULT, step=1),
        slider("conv-padding", "Zero-padding P", 0, 3, helpers.PADDING_DEFAULT, step=1),
    )
