"""Observations (practical notes) for AFML Ch. 3 dashboard pages."""

from __future__ import annotations

TRIPLE_BARRIER = [
    (
        "Fixed-horizon bias",
        r"Fixed-horizon labels that ignore intermediate stops introduce look-ahead and optimistic bias. "
        r"Triple-barrier labels reflect the actual path-dependent exit.",
    ),
]

META_LABELING = [
    (
        "Precision-recall tradeoff",
        r"Meta-labeling improves precision by abstaining on low-confidence primary signals, "
        r"at the cost of fewer trades. Bet size can scale with meta-model probability.",
    ),
]
