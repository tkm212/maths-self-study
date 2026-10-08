"""Filter controls for the early stopping page."""

from __future__ import annotations

import dl_ch07_helpers as helpers
from dash import html

from maths_self_study.dashboards.components import filter_bar, slider
from maths_self_study.math.regularization import REGRESSION_EPOCHS


def build_filters() -> html.Div:
    return filter_bar(
        slider(
            "es-stop-epoch",
            "Stop epoch",
            20,
            REGRESSION_EPOCHS,
            helpers.STOP_EPOCH_DEFAULT,
            step=10,
        ),
    )
