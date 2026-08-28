"""Definitions for AFML Ch. 2 (Financial Data Structures) dashboard pages."""

from __future__ import annotations

BAR_TYPES = [
    (
        "Information-driven bar",
        r"A bar sampled when a fixed amount of *information* arrives (e.g. fixed tick count, volume, or dollar value), "
        r"rather than at fixed clock time. Returns tend to be closer to IID than time bars.",
    ),
    (
        "Dollar bar",
        r"A bar closes when cumulative traded dollar value $\sum P_t Q_t$ reaches a threshold. "
        r"Normalises sampling by information flow rather than calendar time.",
    ),
]

CUSUM = [
    (
        "CUSUM filter",
        r"A quality-control statistic $S_t$ accumulates signed log-return divergences from zero. "
        r"An event fires when $|S_t| \ge h$, then $S_t$ resets. Avoids repeated triggers while price hovers near a level.",
    ),
    (
        "Event sampling",
        r"Selecting observation times (e.g. via CUSUM) before labeling, instead of labeling every bar. "
        r"Reduces overlap and focuses on meaningful price moves.",
    ),
]

PCA_WEIGHTS = [
    (
        "Principal component loading",
        r"An eigenvector of the return correlation matrix. The first PC captures the dominant shared variation "
        r"across features (here, multi-horizon returns).",
    ),
    (
        "Multi-horizon return",
        r"Returns over different lookback windows on the same asset (e.g. 1, 5, 10, 30 bars). "
        r"Used as a feature matrix for PCA-based composite signals.",
    ),
]
