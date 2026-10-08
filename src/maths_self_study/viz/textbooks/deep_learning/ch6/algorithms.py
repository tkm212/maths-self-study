"""Algorithms for Deep Learning Ch. 6 (Deep Feedforward Networks)."""

from __future__ import annotations

XOR_BACKPROP = (
    "Train XOR with backprop on a one-hidden-layer MLP",
    [
        "Initialize weights for layers $W^{(1)}, W^{(2)}$ and biases $b^{(1)}, b^{(2)}$.",
        "Forward pass: $h = g(W^{(1)} x + b^{(1)})$, $\\hat{y} = \\sigma(W^{(2)} h + b^{(2)})$.",
        "Compute loss $L = \\frac{1}{4}\\sum_i (\\hat{y}_i - y_i)^2$ on the four XOR patterns.",
        "Backward pass: propagate $\\partial L / \\partial \\hat{y}$ through sigmoid and hidden activation.",
        "Update all parameters with gradient descent until XOR is classified correctly.",
    ],
)

BACKPROP = (
    "General back-propagation (Algorithm 6.5)",
    [
        "Run a forward pass and store intermediate values for each node in the graph.",
        "Compute the loss gradient at the output nodes.",
        "Traverse the graph in reverse topological order.",
        "At each node, apply the chain rule to route gradients to parent nodes.",
        "Accumulate gradients for shared inputs; use them in a parameter update rule.",
    ],
)
