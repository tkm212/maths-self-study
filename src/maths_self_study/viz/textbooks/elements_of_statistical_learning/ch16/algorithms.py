"""Algorithms for ESL Ch. 16 dashboard pages."""

from __future__ import annotations

ENSEMBLE_LEARNING = (
    "Stacking with cross-validation",
    [
        r"Train $M$ diverse base models on the full training set.",
        r"For each CV fold, collect out-of-fold predictions $Z_m(x_i)$ for every base $m$.",
        r"Fit meta-learner $h$ on the matrix of OOF meta-features $(Z_1, \ldots, Z_M)$.",
        r"At test time, pass base predictions through the fitted meta-model (ESL §16.2).",
    ],
)
