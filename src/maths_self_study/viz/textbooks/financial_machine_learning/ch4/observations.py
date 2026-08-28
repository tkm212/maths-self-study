"""Observations (practical notes) for AFML Ch. 4 dashboard pages."""

from __future__ import annotations

CONCURRENCY = [
    (
        "Uniqueness weighting",
        r"Weighting each event by average $1/c(t)$ over its lifetime down-weights periods "
        r"where many labels share the same price information.",
    ),
]

SAMPLE_WEIGHTS = [
    (
        "Combined weight",
        r"A common practice multiplies uniqueness-based weights with time-decay factors, then normalizes. "
        r"Pass the result as `sample_weight` to sklearn estimators that support it.",
    ),
]
