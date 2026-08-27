"""Definitions for Deep Learning Ch. 5 (Machine Learning Basics) dashboard pages."""

from __future__ import annotations

CAPACITY = [
    (
        "Model capacity",
        r"The richness of functions a model can represent. "
        r"Higher-capacity models can fit more complex patterns — and more noise.",
    ),
    (
        "Overfitting",
        r"When training error is low but test error is high because the model memorised idiosyncrasies of the training set.",
    ),
]

VALIDATION = [
    (
        "Validation set",
        r"Held-out data used to estimate generalisation while tuning hyperparameters — "
        r"never used for gradient updates, only for model selection.",
    ),
    (
        "Generalisation gap",
        r"The difference between training and validation error; a widening gap signals overfitting.",
    ),
]

BIAS_VARIANCE = [
    (
        "Bias",
        r"Error from overly rigid models that systematically miss structure in the data (underfitting).",
    ),
    (
        "Variance",
        r"Error from models that fit random fluctuations in the training set; high variance worsens test performance.",
    ),
]

MLE = [
    (
        "Likelihood",
        r"$L(\theta) = \prod_i p_{\mathrm{model}}(x^{(i)}; \theta)$ is the probability of the observed data given parameters $\theta$.",
    ),
    (
        "Maximum likelihood estimate",
        r"The parameter value $\hat{\theta}$ that maximises $L(\theta)$ — equivalently, that makes the observed data most probable under the model.",
    ),
]

SGD = [
    (
        "Mini-batch",
        r"A random subset $B$ of training examples used to estimate the gradient each step; "
        r"cheaper than full-batch GD and less noisy than batch size $1$.",
    ),
    (
        "Stochastic gradient descent",
        r"An iterative update $\theta \leftarrow \theta - \eta \nabla_\theta L$ using a gradient estimated from a mini-batch rather than the full dataset.",
    ),
]
