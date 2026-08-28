"""Definitions for AFML Ch. 3 (Labeling) dashboard pages."""

from __future__ import annotations

TRIPLE_BARRIER = [
    (
        "Triple-barrier label",
        r"For event at $t_0$, three barriers compete: upper (profit take at $P_{t_0}(1+\mathit{pt})$), "
        r"lower (stop loss at $P_{t_0}(1-\mathit{sl})$), and vertical (max hold $N$ bars). "
        r"Label $+1$, $-1$, or $0$ according to which is hit first.",
    ),
    (
        "Path-dependent outcome",
        r"The realized label depends on the price path between entry and exit, not a fixed-horizon return. "
        r"Matches how trades are actually closed in live trading.",
    ),
]

META_LABELING = [
    (
        "Meta-label",
        r"A secondary binary label: take the bet (1) or pass (0), given a primary model's side signal. "
        r"Separates direction from sizing and filters false positives.",
    ),
    (
        "Primary vs secondary model",
        r"The primary model decides *when* and *which side*; the meta-model decides *whether* to act and *how much*. "
        r"Triple-barrier outcomes on primary events train the meta classifier.",
    ),
]
