"""Filter controls for the least squares page."""

from __future__ import annotations

from dash import html

from maths_self_study.dashboards.components import filter_bar, num_input_row, section
from maths_self_study.demos.deep_learning import ch4 as helpers


def build_filters() -> html.Div:
    y = helpers.LS_TARGETS
    items = [(f"y{i}", float(value)) for i, value in enumerate(y)]
    return filter_bar(
        section("Targets b", *num_input_row("ls-y", items)),
    )
