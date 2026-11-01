"""Filter controls for the unfolding page."""

from __future__ import annotations

import dl_ch10_helpers as helpers
from dash import html

from maths_self_study.dashboards.components import filter_bar, slider


def build_filters() -> html.Div:
    return filter_bar(
        slider("unfold-theta", "Recurrence θ", -1.2, 1.4, helpers.THETA_DEFAULT, step=0.05),
        slider("unfold-gain", "Input gain", 0.0, 2.0, helpers.INPUT_GAIN_DEFAULT, step=0.1),
        slider("unfold-pulse", "Pulse time", 1, 18, helpers.PULSE_DEFAULT, step=1),
    )
