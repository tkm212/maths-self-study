"""Definitions for ESL Ch. 13 dashboard pages."""

from __future__ import annotations

PROTOTYPE_METHODS = [
    (
        "Prototype classifier",
        r"$C(x) = \arg\min_g \min_r \|x - m_{g,r}\|$ where $m_{g,r}$ are class prototypes (ESL §13.2).",
    ),
    (
        "K-means prototypes",
        r"Run K-means with $K=R$ separately on each class to obtain $R$ prototypes per class (ESL §13.2.1).",
    ),
    (
        "LVQ",
        r"Learning Vector Quantization moves prototypes toward correctly classified points and away from misclassified ones (ESL §13.2.2).",
    ),
]

NEAREST_NEIGHBORS = [
    (
        "k-NN rule",
        r"$\hat{C}(x) = \arg\max_g \frac{1}{k}\sum_{i \in \mathcal{N}_k(x)} \mathbf{1}(y_i = g)$ under a chosen distance metric (ESL §13.3).",
    ),
    (
        "Effective parameters",
        r"k-NN has roughly $N/k$ effective parameters: $k=1$ memorises; $k=N$ is the global majority vote (ESL §13.3).",
    ),
    (
        "Cover-Hart bound",
        r"As $N \to \infty$, the 1-NN error satisfies $R^* \leq R_{1\text{NN}} \leq 2R^*$ (ESL §13.3).",
    ),
]
