"""Theorems for ESL Ch. 16 dashboard pages."""

from __future__ import annotations

ENSEMBLE_LEARNING = [
    (
        "Variance reduction by averaging",
        r"$\mathrm{Var}(\bar{T}) = \rho \sigma^2 + (1-\rho)\sigma^2/M$; ensembles gain when base errors are weakly correlated (ESL §16.1, Ch. 8).",
    ),
    (
        "Stacking generalisation",
        r"Meta-learner fit on OOF base outputs avoids in-sample leakage; learned weights adapt to joint base behaviour (ESL §16.2).",
    ),
]
