"""Theorems for ESL Ch. 8 dashboard pages."""

from __future__ import annotations

EM_ALGORITHM = [
    (
        "EM monotonicity",
        r"Each EM iteration increases (or leaves unchanged) the observed-data log-likelihood — convergence is to a local maximum (ESL §8.5).",
    ),
]

BAGGING = [
    (
        "Variance reduction",
        r"Averaging $B$ bootstrap fits reduces variance; with uncorrelated resamples, ensemble variance scales as $\sigma^2/B$ (ESL §8.7).",
    ),
]
