"""Theorems for Deep Learning Ch. 3 (Probability and Information Theory) dashboard pages."""

from __future__ import annotations

RANDOM_VARIABLES = [
    (
        "Linearity of expectation",
        "E[aX + bY] = aE[X] + bE[Y] for constants a, b and random variables X, Y, regardless of dependence.",
    ),
    (
        "Law of total probability",
        "If {Bᵢ} partitions the sample space, then P(A) = Σᵢ P(A | Bᵢ)P(Bᵢ).",
    ),
]

DISTRIBUTIONS = [
    (
        "Maximum Bernoulli entropy",
        "Among Bernoulli(p) distributions, Shannon entropy H(p) is maximised at p = ½.",
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
    (
        "Entropy upper bound",
        "For a random variable on n outcomes, H(P) ≤ log n, with equality for the uniform distribution.",
    ),
]

MARKOV = [
    (
        "Chain rule of probability",
        "For any collection of random variables, P(x₁, …, xₙ) = P(x₁) Πᵢ₌₂ⁿ P(xᵢ | x₁, …, xᵢ₋₁).",
    ),
]
