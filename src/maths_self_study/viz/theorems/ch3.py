"""Theorems for Deep Learning Ch. 3 (Probability and Information Theory) dashboard pages."""

from __future__ import annotations

RANDOM_VARIABLES = [
    (
        "Linearity of expectation",
        "E[aX + bY] = aE[X] + bE[Y] for constants a, b and random variables X, Y, regardless of dependence.",
    ),
]

BAYES = [
    (
        "Bayes' theorem",
        "For hypotheses H and evidence E with P(E) > 0, P(H | E) = P(E | H)P(H)/P(E).",
    ),
]

INFORMATION = [
    (
        "Gibbs' inequality",
        "For discrete distributions P and Q on the same support, D_KL(P ‖ Q) ≥ 0, with equality iff P = Q.",
    ),
]

MARKOV = [
    (
        "Chain rule of probability",
        "For any collection of random variables, P(x₁, …, xₙ) = P(x₁) Πᵢ₌₂ⁿ P(xᵢ | x₁, …, xᵢ₋₁).",
    ),
]
