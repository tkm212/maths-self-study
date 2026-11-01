"""Filter controls for the RNN page."""

from __future__ import annotations

import dl_ch10_helpers as helpers
from dash import html

from maths_self_study.dashboards.components import filter_bar, slider


def build_filters() -> html.Div:
    return filter_bar(
        slider("rnn-weight", "Recurrence weight", 0.1, 1.3, helpers.RECURRENCE_DEFAULT, step=0.05),
        slider("rnn-cue", "Cue time", 1, 18, helpers.CUE_DEFAULT, step=1),
    )
