"""Filter controls for the output units page."""

from __future__ import annotations

import dl_ch06_helpers as helpers
from dash import html

from maths_self_study.dashboards.components import filter_bar, slider


def build_filters() -> html.Div:
    return filter_bar(
        slider("out-x", "Sigmoid input x", -4.0, 4.0, 0.0, step=0.1),
        slider("out-z0", "Softmax logit class 0", -3.0, 3.0, helpers.SOFTMAX_LOGITS[0], step=0.1),
        slider("out-z1", "Softmax logit class 1", -3.0, 3.0, helpers.SOFTMAX_LOGITS[1], step=0.1),
        slider("out-z2", "Softmax logit class 2", -3.0, 3.0, helpers.SOFTMAX_LOGITS[2], step=0.1),
    )
