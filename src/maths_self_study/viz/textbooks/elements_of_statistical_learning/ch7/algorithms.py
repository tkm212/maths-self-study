"""Algorithms for ESL Ch. 7 dashboard pages."""

from __future__ import annotations

CROSS_VALIDATION = (
    "K-fold cross-validation",
    [
        r"Partition data into $K$ disjoint folds $\mathcal{F}_1, \ldots, \mathcal{F}_K$ of roughly equal size.",
        r"For each fold $k$: train on $\mathcal{F}_{-k} = \mathcal{D} \setminus \mathcal{F}_k$, score on held-out fold $\mathcal{F}_k$.",
        r"Compute $\mathrm{Err}_k$ on fold $k$; average $\mathrm{CV}(K) = \frac{1}{K}\sum_{k=1}^K \mathrm{Err}_k$.",
        r"Select the model/hyperparameter with lowest $\mathrm{CV}(K)$ (ESL §7.10).",
    ],
)
