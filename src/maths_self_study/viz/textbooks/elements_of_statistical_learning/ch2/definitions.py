"""Definitions for ESL Ch. 2 (Supervised Learning) dashboard pages."""

from __future__ import annotations

K_NEAREST_NEIGHBORS = [
    (
        "k-nearest-neighbour regression",
        r"At input $x_0$, predict $\hat{f}(x_0) = \frac{1}{k}\sum_{x_i \in \mathcal{N}_k(x_0)} y_i$, "
        r"the average response among the $k$ closest training points.",
    ),
    (
        "Local regression",
        r"A model that adapts to the neighbourhood of each query point rather than fitting one global function. "
        r"k-NN is the simplest local method; kernel smoothers (Ch. 6) generalise it with weighted averages.",
    ),
    (
        "Bias-variance trade-off in k",
        r"As $k \downarrow$, variance rises and bias falls; as $k \uparrow$, bias rises and variance falls. "
        r"Test error is U-shaped in $k$ with a minimum between the two extremes (ESL §2.4).",
    ),
]

LEAST_SQUARES = [
    (
        "Ordinary least squares (OLS)",
        r"Choose $\hat{\beta}$ to minimise $\|y - X\beta\|_2^2$. When $X^\top X$ is invertible, "
        r"$\hat{\beta} = (X^\top X)^{-1} X^\top y$ — a closed-form linear estimator with no tuning parameter.",
    ),
    (
        "Training vs test error",
        r"Training error $\overline{\text{err}}$ is computed on data used to fit the model; test error estimates "
        r"performance on new $(X, Y)$ pairs. Their gap measures optimism from overfitting.",
    ),
]
