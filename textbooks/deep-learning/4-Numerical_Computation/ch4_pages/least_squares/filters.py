"""Filter controls for the least squares page."""

from __future__ import annotations

from dash import html

from maths_self_study.dashboards.components import filter_bar, num_input, section
from maths_self_study.deep_learning import ch4_helpers as helpers


def build_filters() -> html.Div:
    y = helpers.LS_TARGETS
    return filter_bar(
        section(
            "Targets b",
            num_input("ls-y0", "y₀", float(y[0]), step=0.1),
            num_input("ls-y1", "y₁", float(y[1]), step=0.1),
            num_input("ls-y2", "y₂", float(y[2]), step=0.1),
            num_input("ls-y3", "y₃", float(y[3]), step=0.1),
        ),
    )
