"""Definitions for ESL Ch. 7 dashboard pages."""

from __future__ import annotations

BIAS_VARIANCE = [
    (
        "Test error",
        r"$\text{Err}_\mathcal{T} = \mathbb{E}[L(Y, \hat{f}(X)) \mid \mathcal{T}]$ — generalisation error for a fixed training set.",
    ),
    (
        "Optimism",
        r"The gap between in-sample and test error; grows with model complexity (ESL §7.4).",
    ),
]

CROSS_VALIDATION = [
    (
        "K-fold CV",
        r"$\text{CV}(K) = \frac{1}{K}\sum_{k=1}^K \text{Err}_k$ — average held-out fold error.",
    ),
    (
        "Mallows' Cp",
        r"$C_p = \overline{\text{err}} + 2d\hat{\sigma}^2/N$ — linear penalty in the number of parameters $d$.",
    ),
    (
        ".632 bootstrap",
        r"$\widehat{\text{Err}}^{.632} = 0.368\,\overline{\text{err}} + 0.632\,\widehat{\text{Err}}^{(1)}$ balances train and OOB bias.",
    ),
]
