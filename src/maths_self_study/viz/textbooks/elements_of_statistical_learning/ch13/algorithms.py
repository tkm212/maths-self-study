"""Algorithms for ESL Ch. 13 dashboard pages."""

from __future__ import annotations

PROTOTYPE_METHODS = (
    "K-means prototype construction",
    [
        r"For each class $g$, collect observations with $y_i = g$.",
        r"Run K-means with $K=R$ on class-$g$ points to obtain prototypes $m_{g,1},\ldots,m_{g,R}$.",
        r"Classify $x$ by nearest prototype across all classes.",
        r"Optional LVQ updates move prototypes to reduce training misclassification (ESL §13.2).",
    ],
)

NEAREST_NEIGHBORS = (
    "k-NN classification",
    [
        r"Standardise features so no single dimension dominates Euclidean distance.",
        r"For each test point $x$, find the $k$ training indices minimising $d(x, x_i)$.",
        r"Assign the majority class among those neighbours.",
        r"Select $k$ by cross-validation on a log-spaced grid (ESL §13.3).",
    ],
)
