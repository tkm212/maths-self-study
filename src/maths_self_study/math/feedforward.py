"""
Feedforward network utilities (Goodfellow, Bengio & Courville, Ch. 6).

Small NumPy building blocks for dashboard demos: activations, softmax, and
training a one-hidden-layer MLP on XOR or a 1D regression curve.
"""

from __future__ import annotations

from typing import Literal

import numpy as np

ActivationName = Literal["relu", "sigmoid", "tanh"]

XOR_INPUTS = np.array([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]], dtype=float)
XOR_TARGETS = np.array([0.0, 1.0, 1.0, 0.0], dtype=float)


def sigmoid(z: np.ndarray) -> np.ndarray:
    z = np.asarray(z, dtype=float)
    return 1.0 / (1.0 + np.exp(-np.clip(z, -500.0, 500.0)))


def relu(z: np.ndarray) -> np.ndarray:
    return np.maximum(0.0, np.asarray(z, dtype=float))


def tanh(z: np.ndarray) -> np.ndarray:
    return np.tanh(np.asarray(z, dtype=float))


def activation_fn(name: ActivationName):
    return {"relu": relu, "sigmoid": sigmoid, "tanh": tanh}[name]


def activation_deriv(name: ActivationName, z: np.ndarray) -> np.ndarray:
    z = np.asarray(z, dtype=float)
    if name == "relu":
        return (z > 0.0).astype(float)
    if name == "sigmoid":
        s = sigmoid(z)
        return s * (1.0 - s)
    s = tanh(z)
    return 1.0 - s**2


def softmax(logits: np.ndarray) -> np.ndarray:
    z = np.asarray(logits, dtype=float).ravel()
    z = z - z.max()
    exp_z = np.exp(z)
    return exp_z / exp_z.sum()


def train_xor_mlp(
    *,
    n_hidden: int = 4,
    learning_rate: float = 0.5,
    n_epochs: int = 5000,
    activation: ActivationName = "tanh",
    seed: int = 1,
) -> dict[str, np.ndarray | float | ActivationName]:
    """Train a 2 → h → 1 MLP on XOR with sigmoid output and MSE loss."""
    rng = np.random.default_rng(seed)
    n_hidden = max(2, int(n_hidden))
    x = XOR_INPUTS
    y = XOR_TARGETS.reshape(-1, 1)
    w1 = rng.normal(0.0, 1.0, size=(2, n_hidden))
    b1 = np.zeros(n_hidden)
    w2 = rng.normal(0.0, 1.0, size=(n_hidden, 1))
    b2 = np.zeros(1)
    act = activation_fn(activation)
    d_act = lambda z: activation_deriv(activation, z)

    for _ in range(int(n_epochs)):
        z1 = x @ w1 + b1
        h = act(z1)
        z2 = h @ w2 + b2
        pred = sigmoid(z2)
        error = pred - y
        dz2 = error * pred * (1.0 - pred)
        dw2 = h.T @ dz2
        db2 = dz2.sum(axis=0)
        dh = dz2 @ w2.T
        dz1 = dh * d_act(z1)
        dw1 = x.T @ dz1
        db1 = dz1.sum(axis=0)
        w2 -= learning_rate * dw2
        b2 -= learning_rate * db2
        w1 -= learning_rate * dw1
        b1 -= learning_rate * db1

    z1 = x @ w1 + b1
    h = act(z1)
    pred = sigmoid(h @ w2 + b2).ravel()
    mse = float(np.mean((pred - XOR_TARGETS) ** 2))
    return {
        "W1": w1,
        "b1": b1,
        "W2": w2,
        "b2": b2,
        "predictions": pred,
        "mse": mse,
        "activation": activation,
    }


def predict_mlp_grid(
    model: dict[str, np.ndarray | float | ActivationName],
    x_grid: np.ndarray,
    y_grid: np.ndarray,
) -> np.ndarray:
    """Decision surface for the trained XOR MLP on a 2D grid."""
    w1 = model["W1"]
    b1 = model["b1"]
    w2 = model["W2"]
    b2 = model["b2"]
    activation = str(model["activation"])
    act = activation_fn(activation)  # type: ignore[arg-type]
    pts = np.column_stack([x_grid.ravel(), y_grid.ravel()])
    h = act(pts @ w1 + b1)
    return sigmoid(h @ w2 + b2).reshape(x_grid.shape)


def train_mlp_1d(
    x: np.ndarray,
    y: np.ndarray,
    *,
    n_hidden: int = 8,
    learning_rate: float = 0.01,
    n_epochs: int = 3000,
    activation: ActivationName = "tanh",
    seed: int = 1,
) -> dict[str, np.ndarray | float | ActivationName]:
    """Fit y ≈ f(x) with a one-hidden-layer MLP (universal approximation demo)."""
    rng = np.random.default_rng(seed)
    xs = np.asarray(x, dtype=float).ravel()
    ys = np.asarray(y, dtype=float).ravel()
    n_hidden = max(2, int(n_hidden))
    x_feat = xs.reshape(-1, 1)
    w1 = rng.normal(0.0, 0.5, size=(1, n_hidden))
    b1 = np.zeros(n_hidden)
    w2 = rng.normal(0.0, 0.5, size=(n_hidden, 1))
    b2 = np.zeros(1)
    act = activation_fn(activation)
    d_act = lambda z: activation_deriv(activation, z)

    for _ in range(int(n_epochs)):
        z1 = x_feat @ w1 + b1
        h = act(z1)
        pred = h @ w2 + b2
        error = pred.ravel() - ys
        dz2 = error.reshape(-1, 1)
        dw2 = h.T @ dz2
        db2 = dz2.sum(axis=0)
        dh = dz2 @ w2.T
        dz1 = dh * d_act(z1)
        dw1 = x_feat.T @ dz1
        db1 = dz1.sum(axis=0)
        w2 -= learning_rate * dw2
        b2 -= learning_rate * db2
        w1 -= learning_rate * dw1
        b1 -= learning_rate * db1

    z1 = x_feat @ w1 + b1
    h = act(z1)
    pred = (h @ w2 + b2).ravel()
    mse = float(np.mean((pred - ys) ** 2))
    return {
        "W1": w1,
        "b1": b1,
        "W2": w2,
        "b2": b2,
        "predictions": pred,
        "mse": mse,
        "activation": activation,
    }


def predict_mlp_1d(model: dict[str, np.ndarray | float | ActivationName], x_line: np.ndarray) -> np.ndarray:
    w1 = model["W1"]
    b1 = model["b1"]
    w2 = model["W2"]
    b2 = model["b2"]
    activation = model["activation"]
    act = activation_fn(activation)  # type: ignore[arg-type]
    x_feat = np.asarray(x_line, dtype=float).ravel().reshape(-1, 1)
    h = act(x_feat @ w1 + b1)
    return (h @ w2 + b2).ravel()
