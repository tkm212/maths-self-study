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

SEMI_SUPERVISED_MULTITASK = [
    (
        "Semi-supervised learning",
        r"Use unlabeled data with an auxiliary objective — here, consistency under "
        r"input noise — so representations generalize with fewer labels (§7.6).",
    ),
    (
        "Multitask learning",
        r"Train related tasks together with a shared representation; inductive "
        r"transfer often lowers validation error on each task (§7.7).",
    ),
]

PARAMETER_SHARING = [
    (
        "Parameter sharing",
        r"Reuse the same weights across locations (e.g. convolution kernels) to "
        r"cut parameter count and encode translation equivariance (§7.9).",
    ),
    (
        "Convolution",
        r"A 1D filter applied at every position detects local patterns with far "
        r"fewer weights than a fully connected layer over the full input.",
    ),
]

BAGGING = [
    (
        "Bagging",
        r"Train multiple models on bootstrap samples of the data and average their "
        r"predictions to reduce variance (§7.11).",
    ),
    (
        "Ensemble variance",
        r"Independent errors partially cancel when models are averaged, improving "
        r"generalization without changing the base architecture.",
    ),
]

ADVERSARIAL = [
    (
        "Adversarial examples",
        r"Small, often imperceptible input perturbations aligned with the loss "
        r"gradient can sharply increase error (§7.13).",
    ),
    (
        "Adversarial training",
        r"Training on perturbed inputs encourages smoothness and robustness in "
        r"high-density regions of the input space.",
    ),
]

TANGENT_DISTANCE = [
    (
        "Tangent distance",
        r"Distance to a template measured along known transformation directions "
        r"(tangents) is invariant to those nuisances (§7.14).",
    ),
    (
        "Tangent propagation",
        r"Linearize a classifier around small transformations so invariance can "
        r"be baked into the decision rule.",
    ),
]
