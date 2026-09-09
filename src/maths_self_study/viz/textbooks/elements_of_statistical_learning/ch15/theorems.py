"""Theorems for ESL Ch. 15 dashboard pages."""

from __future__ import annotations

RANDOM_FORESTS = [
    (
        "Breiman variance bound",
        r"$\mathrm{Var}(\bar{T}) = \rho \sigma^2 + (1-\rho)\sigma^2/B$; reducing tree correlation $\rho$ via random feature subsampling is the main variance lever (ESL §15.4.1).",
    ),
    (
        "Breiman generalisation bound",
        r"$PE^* \leq \bar{\rho} \cdot (1 - s^2)/s^2$ where $s$ is mean tree strength and $\bar{\rho}$ mean pairwise correlation (ESL §15.4.1).",
    ),
]
