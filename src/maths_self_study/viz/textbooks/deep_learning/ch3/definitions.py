"""Definitions for Deep Learning Ch. 3 (Probability and Information Theory) dashboard pages."""

from __future__ import annotations

RANDOM_VARIABLES = [
    (
        "Random variable",
        r"A variable whose value is an outcome of a random process. "
        r"A discrete RV is described by $P(X = x)$ on a finite or countable support.",
    ),
    (
        "Conditional probability",
        r"$P(A \mid B) = P(A, B)/P(B)$ updates beliefs about $A$ once event $B$ is known. "
        r"It is not symmetric: $P(\text{rain} \mid \text{traffic}) \neq P(\text{traffic} \mid \text{rain})$ in general.",
    ),
]

DISTRIBUTIONS = [
    (
        "Probability mass function",
        r"For discrete $X$, $p(x) = P(X = x)$ with $p(x) \ge 0$ and $\sum_x p(x) = 1$.",
    ),
    (
        "Multivariate Gaussian",
        r"A continuous distribution on $\mathbb{R}^n$ specified by mean $\mu$ and covariance $\Sigma$. "
        r"Contours of equal density are ellipses aligned with eigenvectors of $\Sigma$.",
    ),
]

BAYES = [
    (
        "Bayes' rule",
        r"$P(h \mid v) = P(v \mid h)P(h)/P(v)$ turns a prior $P(h)$ and likelihood $P(v \mid h)$ into a posterior after observing $v$.",
    ),
    (
        "Base rate",
        r"The prior $P(h)$ can dominate the posterior when data are scarce or the likelihood is weak — "
        r"a rare disease stays unlikely even after a positive test.",
    ),
]

INFORMATION = [
    (
        "Shannon entropy",
        r"$H(P) = -\sum_x P(x) \log P(x)$ measures average surprise in bits (or nats). "
        r"Uniform distributions maximise entropy on a fixed support.",
    ),
    (
        "KL divergence",
        r"$D_{\mathrm{KL}}(P \| Q) = \sum_x P(x) \log(P(x)/Q(x))$ is asymmetric: it penalises $Q$ for assigning low mass where $P$ is high.",
    ),
]

MARKOV = [
    (
        "Markov property",
        r"A process is Markov if the future depends on the past only through the present: "
        r"$P(x_t \mid x_{<t}) = P(x_t \mid x_{t-1})$.",
    ),
    (
        "Chain rule of probability",
        r"Any joint distribution factorises as a product of conditionals, e.g. "
        r"$P(x_1, x_2, x_3) = P(x_1)P(x_2 \mid x_1)P(x_3 \mid x_2)$ for a chain graph.",
    ),
]
