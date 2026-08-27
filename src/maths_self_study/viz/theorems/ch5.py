"""Theorems for Deep Learning Ch. 5 (Machine Learning Basics) dashboard pages."""

from __future__ import annotations

CAPACITY = [
    (
        "Interpolation with sufficient capacity",
        "A polynomial of degree at most n − 1 can interpolate any n distinct points exactly; "
        "higher degree than necessary can fit noise as well as signal.",
    ),
]

VALIDATION = [
    (
        "Empirical risk vs true risk",
        "Training error R̂_S(f) is computed on sample S; validation error estimates out-of-sample risk R(f). "
        "Minimising R̂ alone does not guarantee low R when capacity is high.",
    ),
]

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

SGD = [
    (
        "Unbiased gradient estimate",
        "A mini-batch gradient (1/|B|)Σ_{i∈B} ∇L(f(x⁽ⁱ⁾), y⁽ⁱ⁾) is an unbiased estimate of the full-batch gradient when B is drawn uniformly.",
    ),
]
