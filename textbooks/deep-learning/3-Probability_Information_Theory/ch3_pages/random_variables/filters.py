"""Filter controls for the random variables page."""

from __future__ import annotations

from dash import html

from maths_self_study.dashboards.components import filter_bar, num_input, section
from maths_self_study.deep_learning import ch3_helpers as helpers


def build_filters() -> html.Div:
    j = helpers.RAIN_TRAFFIC_JOINT
    return filter_bar(
        section(
            "P(weather, traffic) joint",
            num_input("rv-j00", "dry · light", float(j[0, 0])),
            num_input("rv-j01", "dry · heavy", float(j[0, 1])),
            num_input("rv-j10", "rain · light", float(j[1, 0])),
            num_input("rv-j11", "rain · heavy", float(j[1, 1])),
        ),
        section(
            "Discrete moments support",
            num_input("rv-p0", "P(0)", 0.1),
            num_input("rv-p1", "P(1)", 0.2),
            num_input("rv-p2", "P(2)", 0.3),
            num_input("rv-p3", "P(3)", 0.4),
        ),
    )
