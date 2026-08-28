"""Theorems and key results for AFML Ch. 3 dashboard pages."""

from __future__ import annotations

TRIPLE_BARRIER = [
    (
        "First-touch rule",
        r"The label is determined by the *first* barrier touched along the path. "
        r"Fixed-horizon labels that ignore intermediate stops introduce look-ahead and optimistic bias.",
    ),
]

META_LABELING = [
    (
        "Precision–recall tradeoff",
        r"Meta-labeling improves precision by abstaining on low-confidence primary signals, "
        r"at the cost of fewer trades. Bet size can scale with meta-model probability.",
    ),
]
