"""Key LaTeX formulas for Deep Learning Ch. 3 (Probability and Information Theory)."""

from __future__ import annotations

# §3.3 — Random variables
MARGINAL = r"P(X=x) = \sum_y P(X=x, Y=y)"
CONDITIONAL = r"P(X=x \mid Y=y) = \frac{P(X=x, Y=y)}{P(Y=y)}"
EXPECTATION = r"\mathbb{E}[X] = \sum_x x\,P(X=x)"
VARIANCE = r"\mathrm{Var}(X) = \mathbb{E}[(X-\mathbb{E}[X])^2] = \mathbb{E}[X^2] - \mathbb{E}[X]^2"

# §3.9 — Common distributions
BERNOULLI_ENTROPY = r"H(p) = -p\log p - (1-p)\log(1-p)"
CATEGORICAL = r"P(X=k) = p_k, \quad \sum_k p_k = 1"
GAUSSIAN_PDF = (
    r"\mathcal{N}(\mu, \sigma^2): \quad p(x) \propto"
    r"\exp\left(-\frac{(x-\mu)^2}{2\sigma^2}\right)"
)
MULTIVARIATE_GAUSSIAN = (
    r"\mathcal{N}(\mu, \Sigma) \text{ has elliptical level sets; eigenvalues of } \Sigma \text{ are axis variances}"
)

# §3.11 — Bayes
BAYES_RULE = r"P(H \mid E) = \frac{P(E \mid H)\,P(H)}{P(E)}"
LAW_OF_TOTAL_PROB = r"P(E) = \sum_H P(E \mid H)\,P(H)"
POSTERIOR_PROPORTIONAL = r"\text{posterior} \propto \text{prior} \times \text{likelihood}"

# §3.10 — Markov chains
CHAIN_RULE = r"P(x_1, \ldots, x_n) = P(x_1) \prod_{i=2}^{n} P(x_i \mid x_1, \ldots, x_{i-1})"
MARKOV_FACTORISATION = r"P(x_i \mid x_1, \ldots, x_{i-1}) = P(x_i \mid x_{i-1})"

# §3.13 — Information theory
SELF_INFORMATION = r"I(x) = -\log P(x)"
SHANNON_ENTROPY = r"H(P) = \mathbb{E}[-\log P(X)] = -\sum_x P(x)\log P(x)"
CROSS_ENTROPY = r"H(P, Q) = \mathbb{E}_P[-\log Q(X)]"
KL_DIVERGENCE = r"D_{\mathrm{KL}}(P \| Q) = \mathbb{E}_P[\log(P/Q)] = H(P, Q) - H(P) \ge 0"
