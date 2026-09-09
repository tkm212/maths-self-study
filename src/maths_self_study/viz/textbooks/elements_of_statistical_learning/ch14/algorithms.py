"""Algorithms for ESL Ch. 14 dashboard pages."""

from __future__ import annotations

CLUSTERING = (
    "Lloyd's K-means",
    [
        r"Initialise $K$ centroids (e.g. random or k-means++).",
        r"Assign each point to its nearest centroid.",
        r"Update each centroid to the mean of assigned points.",
        r"Repeat until assignments stabilise (ESL §14.3.6).",
    ],
)

PRINCIPAL_COMPONENTS = (
    "PCA via SVD",
    [
        r"Centre and optionally scale the data matrix $X$.",
        r"Compute SVD $X = UDV^\top$; columns of $V$ are principal directions.",
        r"Scores are $Z = XV$; PVE comes from squared singular values.",
        r"For NMF, alternate multiplicative updates on $W \geq 0$ and $H \geq 0$ (ESL §14.5-14.6).",
    ],
)
