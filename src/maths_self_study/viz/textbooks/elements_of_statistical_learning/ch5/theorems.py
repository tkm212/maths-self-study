"""Theorems for ESL Ch. 5 (Basis Expansions and Regularization) dashboard pages."""

from __future__ import annotations

SPLINES = [
    (
        "Spline representation (ESL §5.2)",
        r"A natural cubic spline with $K$ interior knots has $K + 4$ free parameters — equivalent to $K$ cubic pieces "
        r"subject to $3(K-1)$ continuity constraints plus two natural boundary conditions.",
    ),
]

SMOOTHING_SPLINES = [
    (
        "Representer theorem for smoothing splines",
        r"The minimiser of the penalised roughness criterion is a natural cubic spline with knots at the data points. "
        r"No need to search over arbitrary smooth functions (ESL §5.4).",
    ),
]
