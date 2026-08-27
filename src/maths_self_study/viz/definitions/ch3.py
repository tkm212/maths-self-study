"""Definitions for Deep Learning Ch. 3 (Probability and Information Theory) dashboard pages."""

from __future__ import annotations

RANDOM_VARIABLES = [
    (
        "Random variable",
        "A variable whose value is an outcome of a random process. "
        "A discrete RV is described by P(X = x) on a finite or countable support.",
    ),
    (
        "Conditional probability",
        "P(A | B) = P(A, B)/P(B) updates beliefs about A once event B is known. "
        "It is not symmetric: P(rain | traffic) ≠ P(traffic | rain) in general.",
    ),
]

DISTRIBUTIONS = [
    (
        "Probability mass function",
        "For discrete X, p(x) = P(X = x) with p(x) ≥ 0 and Σₓ p(x) = 1.",
    ),
    (
        "Multivariate Gaussian",
        "A continuous distribution on ℝⁿ specified by mean μ and covariance Σ. "
        "Contours of equal density are ellipses aligned with eigenvectors of Σ.",
    ),
]

BAYES = [
    (
        "Bayes' rule",
        "P(h | v) = P(v | h)P(h)/P(v) turns a prior P(h) and likelihood P(v | h) into a posterior after observing v.",
    ),
    (
        "Base rate",
        "The prior P(h) can dominate the posterior when data are scarce or the likelihood is weak — "
        "a rare disease stays unlikely even after a positive test.",
    ),
]

INFORMATION = [
    (
        "Shannon entropy",
        "H(P) = −Σₓ P(x) log P(x) measures average surprise in bits (or nats). "
        "Uniform distributions maximise entropy on a fixed support.",
    ),
    (
        "KL divergence",
        "D_KL(P ‖ Q) = Σₓ P(x) log(P(x)/Q(x)) is asymmetric: it penalises Q for assigning low mass where P is high.",
    ),
]

MARKOV = [
    (
        "Markov property",
        "A process is Markov if the future depends on the past only through the present: "
        "P(x_t | x_{<t}) = P(x_t | x_{t−1}).",
    ),
    (
        "Chain rule of probability",
        "Any joint distribution factorises as a product of conditionals, e.g. "
        "P(x₁, x₂, x₃) = P(x₁)P(x₂ | x₁)P(x₃ | x₂) for a chain graph.",
    ),
]
