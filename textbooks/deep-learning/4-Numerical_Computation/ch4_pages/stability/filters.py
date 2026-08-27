"""Filter controls for the stability page."""

from __future__ import annotations

from dash import html

from maths_self_study.dashboards.components import filter_bar, num_input_row, section
from maths_self_study.deep_learning import ch4_helpers as helpers


def build_filters() -> html.Div:
    z = helpers.SOFTMAX_LOGITS
    items = [(f"z{i}", float(value)) for i, value in enumerate(z)]
    return filter_bar(
        section("Logits z", *num_input_row("stab-z", items, step=10.0)),
    )
