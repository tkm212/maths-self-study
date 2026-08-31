"""Definitions for ESL Ch. 16 dashboard pages."""

from __future__ import annotations

ENSEMBLE_LEARNING = [
    (
        "Stacking",
        r"Level-1 model $g$ combines base outputs: $\hat{g}(x) = h(Z_1(x), \ldots, Z_M(x))$ trained on out-of-fold predictions (ESL §16.2).",
    ),
    (
        "Soft voting",
        r"Average predicted class probabilities from base models with uniform weights - simpler than learned stacking (ESL §16.1).",
    ),
    (
        "Base error diversity",
        r"Mistake indicators $e_m(i) = \mathbf{1}[\hat{y}_m(i) \neq y_i]$; low correlation among $(e_1, \ldots, e_M)$ implies complementary errors (ESL §16.1).",
    ),
]
