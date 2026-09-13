"""Definitions for Deep Learning Ch. 6 (Deep Feedforward Networks) dashboard pages."""

from __future__ import annotations

XOR = [
    (
        "Feedforward network",
        r"A composition of functions with no cycles: information flows from input "
        r"$x$ through hidden layers to output $y = f(x; \theta)$.",
    ),
    (
        "Hidden layer",
        r"Intermediate representation $h = g(Wx + b)$ that lets the network build "
        r"nonlinear features before the output layer.",
    ),
]

ACTIVATIONS = [
    (
        "Rectified linear unit (ReLU)",
        r"$\mathrm{ReLU}(z) = \max(0, z)$ — sparse activations, fast to compute; "
        r"the default choice in most modern deep networks.",
    ),
    (
        "Logistic sigmoid",
        r"$\sigma(z) = 1/(1+e^{-z})$ — squashes to $(0,1)$; common for binary outputs "
        r"and historically for hidden units.",
    ),
]

OUTPUT_UNITS = [
    (
        "Softmax",
        r"Maps a vector of logits $z$ to a valid probability vector: "
        r"$\mathrm{softmax}(z)_i = e^{z_i} / \sum_j e^{z_j}$.",
    ),
    (
        "Bernoulli output",
        r"A single sigmoid unit models $P(y=1 \mid x)$ for binary classification.",
    ),
]

BACKPROP = [
    (
        "Computational graph",
        r"A DAG whose nodes are variables and edges carry operations. Forward "
        r"propagation evaluates node values; reverse-mode AD walks backward with "
        r"the chain rule (§6.5.1).",
    ),
    (
        "Reverse-mode differentiation",
        r"Backprop computes all parameter gradients in one backward pass — "
        r"efficient when there are many inputs (weights) and few outputs (loss).",
    ),
]

UNIVERSAL_APPROX = [
    (
        "Universal approximation",
        r"A feedforward network with one sufficiently wide hidden layer can approximate "
        r"any continuous function on a compact domain to arbitrary accuracy.",
    ),
    (
        "Depth vs width",
        r"Deeper networks can represent some functions with exponentially fewer units "
        r"than shallow ones — depth trades composition for width.",
    ),
]
