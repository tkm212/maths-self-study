"""Filter controls for the information theory page."""

from __future__ import annotations

from dash import html

from maths_self_study.dashboards.components import filter_bar, num_input, section
from maths_self_study.deep_learning import ch3_helpers as helpers


def build_filters() -> html.Div:
    p, q = helpers.INFO_P, helpers.INFO_Q
    return filter_bar(
        section(
            "P (true)",
            num_input("info-p0", "P(a)", float(p[0])),
            num_input("info-p1", "P(b)", float(p[1])),
            num_input("info-p2", "P(c)", float(p[2])),
            num_input("info-p3", "P(d)", float(p[3])),
        ),
        section(
            "Q (model)",
            num_input("info-q0", "Q(a)", float(q[0])),
            num_input("info-q1", "Q(b)", float(q[1])),
            num_input("info-q2", "Q(c)", float(q[2])),
            num_input("info-q3", "Q(d)", float(q[3])),
        ),
    )
