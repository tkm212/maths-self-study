"""
Feedforward network utilities (Goodfellow, Bengio & Courville, Ch. 6).

Small NumPy building blocks for dashboard demos: activations, softmax, and
training a one-hidden-layer MLP on XOR or a 1D regression curve.
"""

from __future__ import annotations

from typing import Literal, TypedDict

import numpy as np

ActivationName = Literal["relu", "sigmoid", "tanh"]


class XorMlpInit(TypedDict):
    W1: np.ndarray
    b1: np.ndarray
    W2: np.ndarray
    b2: np.ndarray
    activation: ActivationName


class XorForwardState(TypedDict):
    z1: np.ndarray
    h: np.ndarray
    z2: float
    pred: float


class BackpropCheckResult(XorMlpInit):
    analytic: dict[str, np.ndarray]
    numeric: dict[str, np.ndarray]
    rel_errors: dict[str, float]
    max_rel_error: float


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


def init_xor_mlp(
    *,
    n_hidden: int = 4,
    activation: ActivationName = "tanh",
    seed: int = 1,
) -> XorMlpInit:
    """Random initial weights for the XOR MLP (used in backprop demos)."""
    rng = np.random.default_rng(seed)
    n_hidden = max(2, int(n_hidden))
    return {
        "W1": rng.normal(0.0, 1.0, size=(2, n_hidden)),
        "b1": np.zeros(n_hidden),
        "W2": rng.normal(0.0, 1.0, size=(n_hidden, 1)),
        "b2": np.zeros(1),
        "activation": activation,
    }


def xor_forward_pass(
    w1: np.ndarray,
    b1: np.ndarray,
    w2: np.ndarray,
    b2: np.ndarray,
    x: np.ndarray,
    *,
    activation: ActivationName,
) -> XorForwardState:
    """Forward pass for one XOR input vector (section 6.5.4)."""
    act = activation_fn(activation)
    x_row = np.asarray(x, dtype=float).reshape(1, -1)
    z1 = x_row @ w1 + b1
    h = act(z1)
    z2 = h @ w2 + b2
    pred = sigmoid(z2)
    return {
        "z1": z1.ravel(),
        "h": h.ravel(),
        "z2": float(z2.ravel()[0]),
        "pred": float(pred.ravel()[0]),
    }


def xor_mlp_backprop(
    w1: np.ndarray,
    b1: np.ndarray,
    w2: np.ndarray,
    b2: np.ndarray,
    *,
    activation: ActivationName,
) -> dict[str, np.ndarray]:
    """Analytical gradients for the XOR MLP on the full batch (matches ``train_xor_mlp``)."""
    x = XOR_INPUTS
    y = XOR_TARGETS.reshape(-1, 1)
    act = activation_fn(activation)
    d_act = lambda z: activation_deriv(activation, z)
    z1 = x @ w1 + b1
    h = act(z1)
    pred = sigmoid(h @ w2 + b2)
    error = pred - y
    # Mean MSE: d/dpred = (2/n) * (pred - y)
    dz2 = (2.0 / x.shape[0]) * error * pred * (1.0 - pred)
    dw2 = h.T @ dz2
    db2 = dz2.sum(axis=0)
    dh = dz2 @ w2.T
    dz1 = dh * d_act(z1)
    dw1 = x.T @ dz1
    db1 = dz1.sum(axis=0)
    return {"W1": dw1, "b1": db1, "W2": dw2, "b2": db2}


def _xor_mlp_loss(
    w1: np.ndarray,
    b1: np.ndarray,
    w2: np.ndarray,
    b2: np.ndarray,
    *,
    activation: ActivationName,
) -> float:
    x = XOR_INPUTS
    y = XOR_TARGETS.reshape(-1, 1)
    act = activation_fn(activation)
    h = act(x @ w1 + b1)
    pred = sigmoid(h @ w2 + b2)
    return float(np.mean((pred - y) ** 2))


def xor_mlp_numerical_gradients(
    w1: np.ndarray,
    b1: np.ndarray,
    w2: np.ndarray,
    b2: np.ndarray,
    *,
    activation: ActivationName,
    epsilon: float = 1e-5,
) -> dict[str, np.ndarray]:
    """Finite-difference gradients for gradient-checking backprop (section 6.5)."""

    def perturb(param: np.ndarray, index: tuple[int, ...], delta: float) -> np.ndarray:
        out = param.copy()
        out[index] += delta
        return out

    eps = float(epsilon)
    num_grads: dict[str, np.ndarray] = {}
    for name, param in (("W1", w1), ("b1", b1), ("W2", w2), ("b2", b2)):
        grad = np.zeros_like(param, dtype=float)
        for index in np.ndindex(param.shape):
            loss_plus = _xor_mlp_loss(
                w1 if name != "W1" else perturb(w1, index, eps),
                b1 if name != "b1" else perturb(b1, index, eps),
                w2 if name != "W2" else perturb(w2, index, eps),
                b2 if name != "b2" else perturb(b2, index, eps),
                activation=activation,
            )
            loss_minus = _xor_mlp_loss(
                w1 if name != "W1" else perturb(w1, index, -eps),
                b1 if name != "b1" else perturb(b1, index, -eps),
                w2 if name != "W2" else perturb(w2, index, -eps),
                b2 if name != "b2" else perturb(b2, index, -eps),
                activation=activation,
            )
            grad[index] = (loss_plus - loss_minus) / (2.0 * eps)
        num_grads[name] = grad
    return num_grads


def backprop_gradient_check(
    *,
    n_hidden: int = 4,
    activation: ActivationName = "tanh",
    epsilon: float = 1e-5,
    seed: int = 1,
) -> BackpropCheckResult:
    """Compare reverse-mode backprop gradients to finite differences."""
    init = init_xor_mlp(n_hidden=n_hidden, activation=activation, seed=seed)
    w1: np.ndarray = init["W1"]
    b1: np.ndarray = init["b1"]
    w2: np.ndarray = init["W2"]
    b2: np.ndarray = init["b2"]
    analytic = xor_mlp_backprop(w1, b1, w2, b2, activation=activation)
    numeric = xor_mlp_numerical_gradients(
        w1,
        b1,
        w2,
        b2,
        activation=activation,
        epsilon=epsilon,
    )
    rel_errors: dict[str, float] = {}
    for key in ("W1", "b1", "W2", "b2"):
        diff = analytic[key] - numeric[key]
        denom = np.maximum(np.abs(numeric[key]), 1e-8)
        rel_errors[key] = float(np.max(np.abs(diff) / denom))
    return BackpropCheckResult(
        W1=w1,
        b1=b1,
        W2=w2,
        b2=b2,
        activation=activation,
        analytic=analytic,
        numeric=numeric,
        rel_errors=rel_errors,
        max_rel_error=float(max(rel_errors.values())),
    )


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
    x_mean = float(xs.mean())
    x_std = float(xs.std()) or 1.0
    y_mean = float(ys.mean())
    y_std = float(ys.std()) or 1.0
    x_norm = (xs - x_mean) / x_std
    y_norm = (ys - y_mean) / y_std
    x_feat = x_norm.reshape(-1, 1)
    w1 = rng.normal(0.0, 0.5, size=(1, n_hidden))
    b1 = np.zeros(n_hidden)
    w2 = rng.normal(0.0, 0.5, size=(n_hidden, 1))
    b2 = np.zeros(1)
    act = activation_fn(activation)
    d_act = lambda z: activation_deriv(activation, z)
    step = learning_rate / np.sqrt(n_hidden)

    for _ in range(int(n_epochs)):
        z1 = x_feat @ w1 + b1
        h = act(z1)
        pred = h @ w2 + b2
        error = pred.ravel() - y_norm
        dz2 = error.reshape(-1, 1)
        dw2 = h.T @ dz2
        db2 = dz2.sum(axis=0)
        dh = dz2 @ w2.T
        dz1 = dh * d_act(z1)
        dw1 = x_feat.T @ dz1
        db1 = dz1.sum(axis=0)
        w2 -= step * dw2
        b2 -= step * db2
        w1 -= step * dw1
        b1 -= step * db1

    z1 = x_feat @ w1 + b1
    h = act(z1)
    pred_norm = (h @ w2 + b2).ravel()
    pred = pred_norm * y_std + y_mean
    mse = float(np.mean((pred - ys) ** 2))
    return {
        "W1": w1,
        "b1": b1,
        "W2": w2,
        "b2": b2,
        "predictions": pred,
        "mse": mse,
        "activation": activation,
        "x_mean": x_mean,
        "x_std": x_std,
        "y_mean": y_mean,
        "y_std": y_std,
    }


def predict_mlp_1d(model: dict[str, np.ndarray | float | ActivationName], x_line: np.ndarray) -> np.ndarray:
    w1 = model["W1"]
    b1 = model["b1"]
    w2 = model["W2"]
    b2 = model["b2"]
    activation = model["activation"]
    act = activation_fn(activation)  # type: ignore[arg-type]
    xs = np.asarray(x_line, dtype=float).ravel()
    x_mean = float(model["x_mean"])
    x_std = float(model["x_std"])
    y_mean = float(model["y_mean"])
    y_std = float(model["y_std"])
    x_feat = ((xs - x_mean) / x_std).reshape(-1, 1)
    h = act(x_feat @ w1 + b1)
    return (h @ w2 + b2).ravel() * y_std + y_mean
