"""Proofs for Deep Learning Ch. 3 (Probability and Information Theory) dashboard pages."""

from __future__ import annotations

BAYES = (
    "Bayes' theorem",
    [
        r"By definition of conditional probability, $P(H \mid E) = P(H, E) / P(E)$ and $P(E \mid H) = P(H, E) / P(H)$.",
        r"Solve for the joint: $P(H, E) = P(E \mid H)\, P(H)$.",
        r"Substitute into the first identity (assuming $P(E) > 0$): "
        r"$P(H \mid E) = P(E \mid H)\, P(H) / P(E)$.",
        r"The denominator normalises over all hypotheses: "
        r"$P(E) = \sum_H P(E \mid H)\, P(H)$ (Goodfellow et al. §3.11).",
    ],
)

GIBBS = (
    "Gibbs' inequality ($D_{\mathrm{KL}} \ge 0$)",
    [
        r"For discrete $P, Q$ on the same support, expand "
        r"$D_{\mathrm{KL}}(P \| Q) = \sum_x P(x) \log \frac{P(x)}{Q(x)}$.",
        r"Use $\log t \le t - 1$ for $t > 0$ (equality iff $t = 1$): "
        r"$\log \frac{P(x)}{Q(x)} \le \frac{P(x)}{Q(x)} - 1$.",
        r"Multiply by $P(x)$ and sum: "
        r"$\sum_x P(x) \log \frac{P(x)}{Q(x)} \le \sum_x P(x) - \sum_x \frac{P(x)^2}{Q(x)}$.",
        r"Since $\sum_x P(x) = 1$ and $\frac{P(x)^2}{Q(x)} \ge P(x)$ by AM–GM (or Jensen on $-\log$), "
        r"the right-hand side is $\le 0$, with equality iff $P(x) = Q(x)$ everywhere.",
        r"Hence $D_{\mathrm{KL}}(P \| Q) \ge 0$, and cross-entropy $H(P,Q) = H(P) + D_{\mathrm{KL}}(P \| Q) \ge H(P)$ "
        r"(Goodfellow et al. §3.13).",
    ],
)
