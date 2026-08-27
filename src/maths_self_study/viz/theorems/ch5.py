"""Theorems for Deep Learning Ch. 5 (Machine Learning Basics) dashboard pages."""

from __future__ import annotations

BIAS_VARIANCE = [
    (
        "Bias-variance decomposition",
        "For squared loss, expected test error decomposes into bias², variance, and irreducible noise: "
        "E[(y − f̂)²] = bias² + variance + σ² under standard assumptions.",
    ),
]

MLE = [
    (
        "Gaussian MLE closed form",
        "For i.i.d. samples from N(μ, σ²), the MLEs are μ̂ = (1/m)Σ x⁽ⁱ⁾ and σ̂² = (1/m)Σ(x⁽ⁱ⁾ − μ̂)².",
    ),
]
