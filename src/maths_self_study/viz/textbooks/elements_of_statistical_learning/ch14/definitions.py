"""Definitions for ESL Ch. 14 dashboard pages."""

from __future__ import annotations

CLUSTERING = [
    (
        "K-means objective",
        r"Minimise WCSS $W(C) = \sum_k \sum_{i \in C_k} \|x_i - m_k\|^2$ via Lloyd's assign/update steps (ESL §14.3.6).",
    ),
    (
        "Silhouette score",
        r"$s_i = (b_i - a_i)/\max(a_i, b_i)$ measures cohesion vs separation for observation $i$ (ESL §14.3.6).",
    ),
    (
        "Linkage",
        r"Agglomerative clustering merges clusters using single, complete, average, or Ward inter-cluster distance (ESL §14.3.12).",
    ),
]

PRINCIPAL_COMPONENTS = [
    (
        "PCA direction",
        r"$v_m = \arg\max_{\|v\|=1,\, v \perp v_1,\ldots,v_{m-1}} \mathrm{Var}(Xv)$ - the $m$-th principal component (ESL §14.5).",
    ),
    (
        "PVE",
        r"Proportion of variance explained: $\mathrm{PVE}_m = \lambda_m / \sum_j \lambda_j$ from the sample covariance eigenvalues.",
    ),
    (
        "NMF",
        r"Factorise $X \approx WH$ with $W, H \geq 0$ for parts-based, interpretable components (ESL §14.6).",
    ),
]
