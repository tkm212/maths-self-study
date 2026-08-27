"""Key LaTeX formulas for Deep Learning Ch. 3 (Probability and Information Theory)."""

from __future__ import annotations

# §3.2–3.3 — Random variables
EXPECTATION = r"\mathbb{E}[X] = \sum_x x \, P(X = x)"
VARIANCE = r"\mathrm{Var}(X) = \mathbb{E}\big[(X - \mathbb{E}[X])^2\big]"
CONDITIONAL_PROB = r"P(A \mid B) = \frac{P(A, B)}{P(B)}"

# §3.9 — Common distributions
GAUSSIAN_PDF = (
    r"\mathcal{N}(x; \mu, \sigma^2) = \frac{1}{\sqrt{2\pi\sigma^2}}"
    r"\exp\left(-\frac{(x-\mu)^2}{2\sigma^2}\right)"
)
BERNOULLI_ENTROPY = r"H(p) = -p \log p - (1-p) \log(1-p)"

# §3.5 — Bayes' rule
BAYES_RULE = r"P(h \mid v) = \frac{P(v \mid h)\, P(h)}{P(v)}"

# §3.13 — Information theory
SHANNON_ENTROPY = r"H(P) = -\sum_x P(x) \log P(x)"
CROSS_ENTROPY = r"H(P, Q) = -\sum_x P(x) \log Q(x)"
KL_DIVERGENCE = r"D_{\mathrm{KL}}(P \| Q) = \sum_x P(x) \log \frac{P(x)}{Q(x)}"

# §3.10 — Structured models
CHAIN_RULE = r"P(x_1, x_2, x_3) = P(x_1)\, P(x_2 \mid x_1)\, P(x_3 \mid x_2)"
