"""Filter controls for the SGD page."""

from __future__ import annotations

from dash import html

from maths_self_study.dashboards.components import filter_bar, slider
from maths_self_study.deep_learning import ch5_helpers as helpers


def build_filters() -> html.Div:
    return filter_bar(
        slider("sgd-eta", "Learning rate eta", 0.01, 0.2, helpers.SGD_LEARNING_RATE, step=0.01),
        slider("sgd-batch", "Mini-batch size", 1, 20, helpers.SGD_BATCH_SIZE, step=1),
    )
