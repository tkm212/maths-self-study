"""Definitions for ESL Ch. 15 dashboard pages."""

from __future__ import annotations

RANDOM_FORESTS = [
    (
        "Random forest",
        r"Average of $B$ trees grown on bootstrap samples with $m$ random features considered at each split (ESL §15.2).",
    ),
    (
        "OOB error",
        r"Out-of-bag prediction averages only trees for which observation $i$ was not in the bootstrap sample - approximates LOO CV (ESL §15.3.1).",
    ),
    (
        "MDI importance",
        r"Mean decrease in impurity: sum weighted split improvements attributable to feature $j$ across all trees (ESL §15.3.2).",
    ),
]
