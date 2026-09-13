"""Filter controls for the least squares page."""

from __future__ import annotations

import dl_ch04_helpers as helpers
from dash import html

from maths_self_study.dashboards.components import filter_bar, num_input_row, section


def build_filters() -> html.Div:
    y = helpers.LS_TARGETS
    items = [(f"y{i}", float(value)) for i, value in enumerate(y)]
    return filter_bar(
        section("Targets b", *num_input_row("ls-y", items)),
    )
