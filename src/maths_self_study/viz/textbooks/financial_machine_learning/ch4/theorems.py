"""Theorems and key results for AFML Ch. 4 dashboard pages."""

from __future__ import annotations

CONCURRENCY = [
    (
        "IID violation from overlap",
        r"Overlapping labels share bars and returns, inflating effective sample size and biasing cross-validation. "
        r"Weighting by $1/c(t)$ corrects for redundant concurrent information.",
    ),
]

SAMPLE_WEIGHTS = [
    (
        "Combined weight",
        r"A common practice multiplies uniqueness-based weights with time-decay factors, then normalizes. "
        r"Pass the result as `sample_weight` to sklearn estimators that support it.",
    ),
]
