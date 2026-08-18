"""Filter controls for the random variables page."""

from __future__ import annotations

from maths_self_study.dashboards.components import filter_bar, num_input, section
from maths_self_study.deep_learning import ch3_helpers as helpers


def build_filters():
    j = helpers.RAIN_TRAFFIC_JOINT
    return filter_bar(
        section(
            "P(weather, traffic) joint — sums to 1",
            num_input("rv-j00", "dry · light", float(j[0, 0]), step=0.05, min_=0.0, max_=1.0),
            num_input("rv-j01", "dry · heavy", float(j[0, 1]), step=0.05, min_=0.0, max_=1.0),
            num_input("rv-j10", "rain · light", float(j[1, 0]), step=0.05, min_=0.0, max_=1.0),
            num_input("rv-j11", "rain · heavy", float(j[1, 1]), step=0.05, min_=0.0, max_=1.0),
        ),
        section(
            "PMF P(X = x) — sums to 1",
            num_input("rv-p0", "P(X=0)", 0.1, step=0.05, min_=0.0, max_=1.0),
            num_input("rv-p1", "P(X=1)", 0.2, step=0.05, min_=0.0, max_=1.0),
            num_input("rv-p2", "P(X=2)", 0.3, step=0.05, min_=0.0, max_=1.0),
            num_input("rv-p3", "P(X=3)", 0.4, step=0.05, min_=0.0, max_=1.0),
        ),
    )
