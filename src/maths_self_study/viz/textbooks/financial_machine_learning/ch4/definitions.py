"""Definitions for AFML Ch. 4 (Sample Weights) dashboard pages."""

from __future__ import annotations

CONCURRENCY = [
    (
        "Label concurrency",
        r"At bar $t$, $c(t)$ is the number of triple-barrier events still open (started but not yet closed). "
        r"High concurrency means many overlapping labels share the same price information.",
    ),
    (
        "Average uniqueness",
        r"For event $i$ spanning $[t_{i,0}, t_{i,1}]$, $\bar{u}_i = \frac{1}{T_i}\sum_{t} \frac{1}{c(t)}$. "
        r"Events that overlap many concurrent labels receive lower uniqueness.",
    ),
]

SAMPLE_WEIGHTS = [
    (
        "Time-decay weight",
        r"Weight $\propto \exp(-\text{age}/\tau)$ with reference time at the end of the sample. "
        r"Discounts stale events relative to the current decision point.",
    ),
    (
        "Sequential bootstrap",
        r"Bagging variant that prefers draws with higher uniqueness, reducing redundancy in each bootstrap replicate. "
        r"Complements explicit per-row `sample_weight` in a single fit.",
    ),
]
