"""Definitions for Deep Learning Ch. 8 dashboard pages."""

from __future__ import annotations

MOMENTUM = [
    (
        "Momentum",
        r"Accumulate a velocity vector $v$ in the direction of recent gradients. "
        r"Dampens oscillation in narrow valleys and speeds progress along shallow directions (§8.3.2).",
    ),
    (
        "Learning rate",
        r"Step size $\eta$ trades off convergence speed against stability; too large diverges, "
        r"too small wastes compute on flat regions.",
    ),
]

INITIALIZATION = [
    (
        "Symmetry breaking",
        r"Random initial weights break unit symmetry so hidden units specialize. "
        r"Scale controls signal variance through layers (§8.4).",
    ),
    (
        "Xavier / Glorot",
        r"Choose variance so activations and gradients neither vanish nor explode in "
        r"tanh/sigmoid networks with roughly linear activations at initialization.",
    ),
]

ADAPTIVE = [
    (
        "Per-parameter learning rates",
        r"AdaGrad, RMSProp, and Adam rescale each parameter's update using running "
        r"statistics of squared gradients (§8.5).",
    ),
    (
        "Adam",
        r"Combines momentum on gradients with adaptive scaling; default choice for "
        r"many deep learning experiments when tuning budget is limited.",
    ),
]

MINIBATCH = [
    (
        "Mini-batch SGD",
        r"Estimate the gradient from a subset of examples each step — cheaper than full "
        r"batch and lower variance than single-example updates (§8.1.3).",
    ),
    (
        "Gradient noise",
        r"Small batches inject noise that can help escape sharp minima; large batches "
        r"give smoother but more expensive steps.",
    ),
]
