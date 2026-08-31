"""Algorithms for ESL Ch. 15 dashboard pages."""

from __future__ import annotations

RANDOM_FORESTS = (
    "Random forest training",
    [
        r"For $b = 1,\ldots,B$: draw bootstrap sample $\mathcal{D}^{*b}$ of size $N$.",
        r"Grow tree $T_b$ to full depth; at each split, consider only $m$ randomly chosen features.",
        r"Predict by averaging tree votes: $\hat{f}_{RF}(x) = \frac{1}{B}\sum_b T_b(x)$.",
        r"Estimate generalisation via OOB error without a separate validation set (ESL §15.2-15.3).",
    ],
)
