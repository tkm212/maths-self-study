"""Algorithms for Deep Learning Ch. 8 dashboard pages."""

from __future__ import annotations

MINIBATCH_SGD = (
    "Mini-batch stochastic gradient descent",
    [
        r"Shuffle the training set (or sample with replacement).",
        r"For each mini-batch of size $m$, compute average loss $J_{\mathrm{batch}}$.",
        r"Update $\theta \leftarrow \theta - \eta \nabla_\theta J_{\mathrm{batch}}$.",
        r"Repeat until validation error stops improving or a step budget is reached.",
    ],
)

ADAM = (
    "Adam optimizer",
    [
        r"Initialize moment estimates $m, v$ to zero and set $t = 0$.",
        r"At step $t$: compute minibatch gradient $g_t$.",
        r"Update biased moments $m_t, v_t$ and bias-correct $\hat{m}_t, \hat{v}_t$ (§8.5).",
        r"Apply $\theta_{t+1} = \theta_t - \eta \hat{m}_t / (\sqrt{\hat{v}_t} + \epsilon)$.",
    ],
)
