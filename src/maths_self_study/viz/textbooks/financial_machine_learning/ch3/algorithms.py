"""Algorithms for AFML Ch. 3 dashboard pages."""

from __future__ import annotations

TRIPLE_BARRIER = (
    "Triple-barrier labeling",
    [
        r"At event time $t_0$ with entry price $P_0$, set upper barrier $U = P_0(1 + \mathrm{pt})$, lower $L = P_0(1 - \mathrm{sl})$, and vertical barrier at $t_0 + \tau$.",
        r"Scan the forward price path $P_t$ for $t > t_0$ until the first barrier touch.",
        r"Label $+1$ if $U$ is touched first, $-1$ if $L$ is touched first, $0$ if the vertical barrier expires first (Snippet 3.2).",
        r"Optional: trigger events with CUSUM filters to reduce label overlap (Ch. 2).",
    ],
)
