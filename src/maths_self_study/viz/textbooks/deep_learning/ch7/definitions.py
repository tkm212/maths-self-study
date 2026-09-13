"""Definitions for Deep Learning Ch. 7 (Regularization) dashboard pages."""

from __future__ import annotations

WEIGHT_DECAY = [
    (
        "Weight decay (L2 penalty)",
        r"Add $\frac{\lambda}{2}\lVert w \rVert_2^2$ to the loss so large weights are "
        r"penalized. Gradient descent shrinks weights toward zero each step.",
    ),
    (
        "Bias-variance tradeoff",
        r"Strong regularization increases bias but reduces variance; validation error "
        r"often has a sweet spot between under- and over-fitting.",
    ),
]

EARLY_STOPPING = [
    (
        "Early stopping",
        r"Stop training when validation error stops improving — a cheap form of "
        r"regularization that limits effective model capacity.",
    ),
    (
        "Patience",
        r"Track the best validation epoch; continuing to train usually lowers "
        r"training error while validation error rises (overfitting).",
    ),
]

DROPOUT = [
    (
        "Dropout",
        r"During training, randomly zero hidden units with probability $p$ and scale "
        r"survivors by $1/(1-p)$. Acts like training an ensemble of thinned networks.",
    ),
    (
        "Inference",
        r"At test time dropout is off; all units contribute. This approximates an "
        r"average over many sub-networks.",
    ),
]

INPUT_NOISE = [
    (
        "Input noise robustness",
        r"Adding noise to inputs during training encourages the network to be "
        r"insensitive to small perturbations — a form of regularization (§7.5).",
    ),
    (
        "Label vs input noise",
        r"Noise on targets encourages output smoothing; noise on inputs encourages "
        r"feature representations robust to measurement error.",
    ),
]
