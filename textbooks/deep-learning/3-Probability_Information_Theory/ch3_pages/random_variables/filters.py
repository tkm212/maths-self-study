"""Filter controls for the random variables page."""

from __future__ import annotations

from maths_self_study.dashboards.components import filter_bar, prob_simplex, section
from maths_self_study.demos.deep_learning import ch3 as helpers


def build_filters():
    j = helpers.RAIN_TRAFFIC_JOINT
    joint_items = [
        ("dry · light", float(j[0, 0])),
        ("dry · heavy", float(j[0, 1])),
        ("rain · light", float(j[1, 0])),
        ("rain · heavy", float(j[1, 1])),
    ]
    pmf_items = [
        ("P(X=0)", 0.1),
        ("P(X=1)", 0.2),
        ("P(X=2)", 0.3),
        ("P(X=3)", 0.4),
    ]
    return filter_bar(
        section(
            "P(weather, traffic) joint — sums to 1",
            *prob_simplex("rv-j", joint_items, id_keys=["00", "01", "10", "11"]),
        ),
        section("PMF P(X = x) — sums to 1", *prob_simplex("rv-p", pmf_items)),
    )
