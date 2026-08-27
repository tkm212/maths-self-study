"""Theorems for Deep Learning Ch. 3 (Probability and Information Theory) dashboard pages."""

from __future__ import annotations

RANDOM_VARIABLES = [
    (
        "Linearity of expectation",
        r"$\mathbb{E}[aX + bY] = a\mathbb{E}[X] + b\mathbb{E}[Y]$ for constants $a, b$ and random variables $X, Y$, regardless of dependence.",
    ),
]

BAYES = [
    (
        "Bayes' theorem",
        r"For hypotheses $H$ and evidence $E$ with $P(E) > 0$, $P(H \mid E) = P(E \mid H)P(H)/P(E)$.",
    ),
]

INFORMATION = [
    (
        "Gibbs' inequality",
        r"For discrete distributions $P$ and $Q$ on the same support, $D_{\mathrm{KL}}(P \| Q) \ge 0$, with equality iff $P = Q$.",
    ),
]

MARKOV = [
    (
        "Chain rule of probability",
        r"For any collection of random variables, $P(x_1, \ldots, x_n) = P(x_1) \prod_{i=2}^{n} P(x_i \mid x_1, \ldots, x_{i-1})$.",
    ),
]
