"""Algorithms for Deep Learning Ch. 5 dashboard pages."""

from __future__ import annotations

SGD = (
    "Stochastic gradient descent",
    [
        r"Initialise parameters $\theta^{(0)}$; choose learning rate $\eta$ and mini-batch size $m$.",
        r"Sample mini-batch $\mathcal{B}_t$ of size $m$ from training data.",
        r"Compute stochastic gradient $\tilde{g}^{(t)} = \frac{1}{m}\sum_{i \in \mathcal{B}_t} \nabla_\theta L(y_i, f(x_i; \theta^{(t)}))$.",
        r"Update $\theta^{(t+1)} = \theta^{(t)} - \eta \tilde{g}^{(t)}$; repeat until validation error stops improving (§5.9).",
    ],
)
