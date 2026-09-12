"""Algorithms for AFML Ch. 2 dashboard pages."""

from __future__ import annotations

CUSUM_FILTER = (
    "CUSUM event filter",
    [
        r"Track cumulative signed log-return $S_t$ from a reset level of zero.",
        r"Sample bar $t$ when $|S_t| \ge h$; reset $S_t \leftarrow 0$ after each event.",
        r"Events mark meaningful shifts — use as labeling seeds instead of every bar.",
        r"Threshold $h$ trades sensitivity against the number of events.",
    ],
)
