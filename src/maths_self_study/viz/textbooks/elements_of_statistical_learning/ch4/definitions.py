"""Definitions for ESL Ch. 4 (Linear Methods for Classification) dashboard pages."""

from __future__ import annotations

LOGISTIC_REGRESSION = [
    (
        "Logistic regression",
        r"Models class log-odds as linear in $x$: "
        r"$\log\frac{P(Y=1 \mid x)}{1 - P(Y=1 \mid x)} = \beta_0 + \beta^\top x$. "
        r"Equivalently $P(Y=1 \mid x) = \sigma(\beta_0 + \beta^\top x)$ with the logistic function $\sigma$.",
    ),
    (
        "Iteratively reweighted least squares (IRLS)",
        r"Newton-Raphson on the concave log-likelihood. Each step solves a weighted least-squares problem — "
        r"the standard way to fit logistic regression (ESL §4.4.1).",
    ),
]

LDA = [
    (
        "Linear discriminant analysis (LDA)",
        r"Assumes $X \mid Y=k \sim \mathcal{N}(\mu_k, \Sigma)$ with a **shared** covariance $\Sigma$. "
        r"The Bayes decision boundary is linear in $x$.",
    ),
    (
        "Quadratic discriminant analysis (QDA)",
        r"Allows class-specific covariances $\Sigma_k$. Decision boundaries are quadratic curves in $x$ — "
        r"more flexible but higher variance than LDA.",
    ),
]

SEPARATING_HYPERPLANES = [
    (
        "Separating hyperplane",
        r"A hyperplane $\{x : \beta^\top x + \beta_0 = 0\}$ that puts each class entirely on one side. "
        r"Exists only when the training data are linearly separable.",
    ),
    (
        "Maximum-margin hyperplane",
        r"The separator that maximises the distance (margin) to the nearest training point. "
        r"For separable data this is the optimal soft/hard-margin SVM solution (ESL §4.5.2).",
    ),
]
