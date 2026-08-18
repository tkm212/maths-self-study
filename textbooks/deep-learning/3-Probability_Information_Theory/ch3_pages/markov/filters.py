"""Filter controls for the Markov / structured models page."""

from __future__ import annotations

from maths_self_study.dashboards.components import filter_bar, prob_pair, section


def build_filters():
    return filter_bar(
        section(
            "P(X₁)",
            prob_pair("mk-p", "P(X₁=0)", "P(X₁=1)", 0.6, 0.4),
        ),
        section(
            "P(X₂ | X₁=0)",
            prob_pair("mk-t0", "P(X₂=0)", "P(X₂=1)", 0.7, 0.3),
        ),
        section(
            "P(X₂ | X₁=1)",
            prob_pair("mk-t1", "P(X₂=0)", "P(X₂=1)", 0.2, 0.8),
        ),
        section(
            "P(X₃ | X₂=0)",
            prob_pair("mk-u0", "P(X₃=0)", "P(X₃=1)", 0.9, 0.1),
        ),
        section(
            "P(X₃ | X₂=1)",
            prob_pair("mk-u1", "P(X₃=0)", "P(X₃=1)", 0.4, 0.6),
        ),
    )
