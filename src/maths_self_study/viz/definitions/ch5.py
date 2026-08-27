"""Definitions for Deep Learning Ch. 5 (Machine Learning Basics) dashboard pages."""

from __future__ import annotations

CAPACITY = [
    (
        "Model capacity",
        "The richness of functions a model can represent. "
        "Higher-capacity models can fit more complex patterns — and more noise.",
    ),
    (
        "Overfitting",
        "When training error is low but test error is high because the model memorised idiosyncrasies of the training set.",
    ),
]

VALIDATION = [
    (
        "Validation set",
        "Held-out data used to estimate generalisation while tuning hyperparameters — "
        "never used for gradient updates, only for model selection.",
    ),
    (
        "Generalisation gap",
        "The difference between training and validation error; a widening gap signals overfitting.",
    ),
]

BIAS_VARIANCE = [
    (
        "Bias",
        "Error from overly rigid models that systematically miss structure in the data (underfitting).",
    ),
    (
        "Variance",
        "Error from models that fit random fluctuations in the training set; high variance worsens test performance.",
    ),
]

MLE = [
    (
        "Likelihood",
        "L(θ) = Πᵢ p_model(x⁽ⁱ⁾; θ) is the probability of the observed data given parameters θ.",
    ),
    (
        "Maximum likelihood estimate",
        "The parameter value θ̂ that maximises L(θ) — equivalently, that makes the observed data most probable under the model.",
    ),
]

SGD = [
    (
        "Mini-batch",
        "A random subset B of training examples used to estimate the gradient each step; "
        "cheaper than full-batch GD and less noisy than batch size 1.",
    ),
    (
        "Stochastic gradient descent",
        "An iterative update θ ← θ − η∇_θ L using a gradient estimated from a mini-batch rather than the full dataset.",
    ),
]
