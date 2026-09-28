"""Filter controls for the semi-supervised / multitask page."""

from __future__ import annotations

import dl_ch07_helpers as helpers
from dash import html

from maths_self_study.dashboards.components import filter_bar, slider


def build_filters() -> html.Div:
    return filter_bar(
        slider("ss-n-labeled", "Labeled training points", 5, 70, helpers.N_LABELED_DEFAULT, step=1),
    )
