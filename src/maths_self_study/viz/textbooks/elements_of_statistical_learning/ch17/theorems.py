"""Theorems for ESL Ch. 17 dashboard pages."""

from __future__ import annotations

GRAPHICAL_MODELS = [
    (
        "Markov property",
        r"Missing edge $(j,k) \notin E$ implies conditional independence of $X_j$ and $X_k$ given the rest (ESL §17.2).",
    ),
    (
        "Precision-sparsity equivalence",
        r"For Gaussian MRFs, $\Theta_{jk} = 0 \Leftrightarrow X_j \perp X_k \mid X_{\setminus \{j,k\}}$ under positivity (ESL §17.3).",
    ),
]
