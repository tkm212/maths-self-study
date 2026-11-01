"""Filter controls for the gated RNN page."""

from __future__ import annotations

import dl_ch10_helpers as helpers
from dash import html

from maths_self_study.dashboards.components import filter_bar, slider


def build_filters() -> html.Div:
    return filter_bar(
        slider("gated-forget", "Forget-gate bias", -3.0, 6.0, helpers.FORGET_DEFAULT, step=0.1),
        slider("gated-cue", "Cue time", 1, 22, helpers.GATED_CUE_DEFAULT, step=1),
        slider("gated-query", "Memory query", 0, 3, helpers.QUERY_DEFAULT, step=1),
    )
