"""
Optimization for training deep models (Goodfellow, Bengio & Courville, Ch. 8).

NumPy demos: momentum on quadratics, weight initialization scales, adaptive
optimizers, and mini-batch noise on a one-hidden-layer MLP regression task.
"""

from __future__ import annotations

from typing import Literal, TypedDict

import numpy as np

from maths_self_study.math.feedforward import ActivationName, activation_deriv, activation_fn
from maths_self_study.math.optimization import gradient_descent_quadratic, quadratic_value


def _normalize_split(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict[str, float]]:
    xs_tr = np.asarray(x_train, dtype=float).ravel()
    ys_tr = np.asarray(y_train, dtype=float).ravel()
    xs_va = np.asarray(x_val, dtype=float).ravel()
    ys_va = np.asarray(y_val, dtype=float).ravel()
    x_mean = float(xs_tr.mean())
    x_std = float(xs_tr.std()) or 1.0
    y_mean = float(ys_tr.mean())
    y_std = float(ys_tr.std()) or 1.0
    stats = {"x_mean": x_mean, "x_std": x_std, "y_mean": y_mean, "y_std": y_std}
    x_tr = (xs_tr - x_mean) / x_std
    y_tr = (ys_tr - y_mean) / y_std
    x_va = (xs_va - x_mean) / x_std
    y_va = (ys_va - y_mean) / y_std
    return x_tr, y_tr, x_va, y_va, stats


def _forward_mlp(
    x_feat: np.ndarray,
    w1: np.ndarray,
    b1: np.ndarray,
    w2: np.ndarray,
    b2: np.ndarray,
    act,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    z1 = x_feat @ w1 + b1
    h = act(z1)
    pred = h @ w2 + b2
    return z1, h, pred


def _mse_norm(pred_norm: np.ndarray, y_norm: np.ndarray) -> float:
    return float(np.mean((pred_norm.ravel() - y_norm.ravel()) ** 2))


def _denorm_predictions(pred_norm: np.ndarray, stats: dict[str, float]) -> np.ndarray:
    return pred_norm * stats["y_std"] + stats["y_mean"]

OptimizerName = Literal["sgd", "momentum", "adam"]


class OptimizerRun(TypedDict):
    val_mse_history: list[float]
    train_mse_history: list[float]
    final_val_mse: float


MOMENTUM_HESSIAN = np.array([[40.0, 0.0], [0.0, 1.0]])
MOMENTUM_LINEAR = np.array([0.0, 0.0])
MOMENTUM_START = np.array([2.5, 2.0])

MLP_HIDDEN = 32
MLP_EPOCHS = 200
MLP_LR = 0.02


def gradient_descent_momentum(
    hessian: np.ndarray,
    gradient_at_start: np.ndarray,
    start: np.ndarray,
    *,
    learning_rate: float,
    momentum: float,
    n_steps: int,
) -> np.ndarray:
    """Heavy-ball momentum on a quadratic (§8.3.2)."""
    h = np.asarray(hessian, dtype=float)
    g = np.asarray(gradient_at_start, dtype=float).ravel()
    x = np.asarray(start, dtype=float).ravel().copy()
    velocity = np.zeros_like(x)
    path = [x.copy()]
    mu = float(np.clip(momentum, 0.0, 0.999))
    for _ in range(n_steps):
        grad = h @ x + g
        velocity = mu * velocity + grad
        x = x - learning_rate * velocity
        path.append(x.copy())
    return np.array(path)


def xavier_std(fan_in: int, fan_out: int) -> float:
    """Glorot uniform scale converted to Gaussian std (§8.4)."""
    fan_in = max(1, int(fan_in))
    fan_out = max(1, int(fan_out))
    limit = np.sqrt(6.0 / (fan_in + fan_out))
    return float(limit / np.sqrt(3.0))


def he_std(fan_in: int) -> float:
    """He init std for ReLU (§8.4)."""
    fan_in = max(1, int(fan_in))
    return float(np.sqrt(2.0 / fan_in))


def momentum_vs_gd_losses(
    *,
    learning_rate: float,
    momentum: float,
    n_steps: int = 25,
) -> tuple[np.ndarray, np.ndarray]:
    """Final loss after each step for vanilla GD and momentum."""
    h = MOMENTUM_HESSIAN
    g = MOMENTUM_LINEAR
    start = MOMENTUM_START
    gd_path = gradient_descent_quadratic(
        h,
        g,
        start,
        learning_rate=learning_rate,
        n_steps=n_steps,
    )
    mom_path = gradient_descent_momentum(
        h,
        g,
        start,
        learning_rate=learning_rate,
        momentum=momentum,
        n_steps=n_steps,
    )
    gd_losses = np.array([quadratic_value(h, g, pt) for pt in gd_path])
    mom_losses = np.array([quadratic_value(h, g, pt) for pt in mom_path])
    return gd_losses, mom_losses


def _init_weights(
    rng: np.random.Generator,
    n_hidden: int,
    *,
    init_scale: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    scale = max(float(init_scale), 1e-8)
    w1 = rng.normal(0.0, scale, size=(1, n_hidden))
    b1 = np.zeros(n_hidden)
    w2 = rng.normal(0.0, scale, size=(n_hidden, 1))
    b2 = np.zeros(1)
    return w1, b1, w2, b2


def initialization_comparison(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    *,
    scale_multiplier: float,
    n_hidden: int = MLP_HIDDEN,
    n_epochs: int = 80,
    activation: ActivationName = "tanh",
    seed: int = 3,
) -> OptimizerRun:
    """Train with init std = multiplier × Xavier (§8.4)."""
    rng = np.random.default_rng(seed)
    n_hidden = max(4, int(n_hidden))
    x_tr, y_tr, x_va, y_va, _stats = _normalize_split(x_train, y_train, x_val, y_val)
    x_feat = x_tr.reshape(-1, 1)
    x_val_feat = x_va.reshape(-1, 1)
    base_std = xavier_std(1, n_hidden)
    init_std = base_std * max(float(scale_multiplier), 1e-4)
    w1, b1, w2, b2 = _init_weights(rng, n_hidden, init_scale=init_std)
    act = activation_fn(activation)
    d_act = lambda z: activation_deriv(activation, z)
    step = MLP_LR / np.sqrt(n_hidden)

    train_hist: list[float] = []
    val_hist: list[float] = []
    for _ in range(int(n_epochs)):
        z1, h, pred = _forward_mlp(x_feat, w1, b1, w2, b2, act)
        error = pred.ravel() - y_tr
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

        _, _, pred_eval = _forward_mlp(x_feat, w1, b1, w2, b2, act)
        _, _, pred_val = _forward_mlp(x_val_feat, w1, b1, w2, b2, act)
        train_hist.append(_mse_norm(pred_eval, y_tr))
        val_hist.append(_mse_norm(pred_val, y_va))

    return {
        "train_mse_history": train_hist,
        "val_mse_history": val_hist,
        "final_val_mse": float(val_hist[-1]),
    }


def _adam_step(
    param: np.ndarray,
    grad: np.ndarray,
    *,
    m: np.ndarray,
    v: np.ndarray,
    t: int,
    lr: float,
    beta1: float,
    beta2: float,
    eps: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    m_new = beta1 * m + (1.0 - beta1) * grad
    v_new = beta2 * v + (1.0 - beta2) * (grad**2)
    m_hat = m_new / (1.0 - beta1**t)
    v_hat = v_new / (1.0 - beta2**t)
    update = lr * m_hat / (np.sqrt(v_hat) + eps)
    return param - update, m_new, v_new


def train_mlp_optimizer(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    *,
    optimizer: OptimizerName = "sgd",
    learning_rate: float = MLP_LR,
    momentum: float = 0.9,
    n_hidden: int = MLP_HIDDEN,
    n_epochs: int = MLP_EPOCHS,
    activation: ActivationName = "relu",
    seed: int = 5,
) -> OptimizerRun:
    """Fit the regression MLP with SGD, momentum, or Adam (§8.3, §8.5)."""
    rng = np.random.default_rng(seed)
    n_hidden = max(4, int(n_hidden))
    x_tr, y_tr, x_va, y_va, stats = _normalize_split(x_train, y_train, x_val, y_val)
    x_feat = x_tr.reshape(-1, 1)
    x_val_feat = x_va.reshape(-1, 1)
    if activation == "relu":
        init_std = he_std(n_hidden)
    else:
        init_std = xavier_std(1, n_hidden)
    w1, b1, w2, b2 = _init_weights(rng, n_hidden, init_scale=init_std)
    act = activation_fn(activation)
    d_act = lambda z: activation_deriv(activation, z)
    base_lr = learning_rate / np.sqrt(n_hidden)
    mu = float(np.clip(momentum, 0.0, 0.999))

    v_w1 = np.zeros_like(w1)
    v_b1 = np.zeros_like(b1)
    v_w2 = np.zeros_like(w2)
    v_b2 = np.zeros_like(b2)

    m_w1 = np.zeros_like(w1)
    m_b1 = np.zeros_like(b1)
    m_w2 = np.zeros_like(w2)
    m_b2 = np.zeros_like(b2)
    v_ad_w1 = np.zeros_like(w1)
    v_ad_b1 = np.zeros_like(b1)
    v_ad_w2 = np.zeros_like(w2)
    v_ad_b2 = np.zeros_like(b2)

    train_hist: list[float] = []
    val_hist: list[float] = []
    adam_t = 0

    for _ in range(int(n_epochs)):
        z1, h, pred = _forward_mlp(x_feat, w1, b1, w2, b2, act)
        error = pred.ravel() - y_tr
        dz2 = error.reshape(-1, 1)
        dw2 = h.T @ dz2
        db2 = dz2.sum(axis=0)
        dh = dz2 @ w2.T
        dz1 = dh * d_act(z1)
        dw1 = x_feat.T @ dz1
        db1 = dz1.sum(axis=0)

        if optimizer == "adam":
            adam_t += 1
            w2, m_w2, v_ad_w2 = _adam_step(
                w2, dw2, m=m_w2, v=v_ad_w2, t=adam_t, lr=base_lr, beta1=0.9, beta2=0.999, eps=1e-8
            )
            b2, m_b2, v_ad_b2 = _adam_step(
                b2, db2, m=m_b2, v=v_ad_b2, t=adam_t, lr=base_lr, beta1=0.9, beta2=0.999, eps=1e-8
            )
            w1, m_w1, v_ad_w1 = _adam_step(
                w1, dw1, m=m_w1, v=v_ad_w1, t=adam_t, lr=base_lr, beta1=0.9, beta2=0.999, eps=1e-8
            )
            b1, m_b1, v_ad_b1 = _adam_step(
                b1, db1, m=m_b1, v=v_ad_b1, t=adam_t, lr=base_lr, beta1=0.9, beta2=0.999, eps=1e-8
            )
        elif optimizer == "momentum":
            v_w2 = mu * v_w2 + dw2
            v_b2 = mu * v_b2 + db2
            v_w1 = mu * v_w1 + dw1
            v_b1 = mu * v_b1 + db1
            w2 -= base_lr * v_w2
            b2 -= base_lr * v_b2
            w1 -= base_lr * v_w1
            b1 -= base_lr * v_b1
        else:
            w2 -= base_lr * dw2
            b2 -= base_lr * db2
            w1 -= base_lr * dw1
            b1 -= base_lr * db1

        _, _, pred_eval = _forward_mlp(x_feat, w1, b1, w2, b2, act)
        _, _, pred_val = _forward_mlp(x_val_feat, w1, b1, w2, b2, act)
        train_hist.append(_mse_norm(pred_eval, y_tr))
        val_hist.append(_mse_norm(pred_val, y_va))

    _, _, pred_val_final = _forward_mlp(x_val_feat, w1, b1, w2, b2, act)
    val_denorm = _denorm_predictions(pred_val_final.ravel(), stats)
    ys_va = np.asarray(y_val, dtype=float).ravel()
    final_val = float(np.mean((val_denorm - ys_va) ** 2))

    return {
        "train_mse_history": train_hist,
        "val_mse_history": val_hist,
        "final_val_mse": final_val,
    }


def minibatch_training_curve(
    x_train: np.ndarray,
    y_train: np.ndarray,
    *,
    batch_size: int,
    n_steps: int = 120,
    learning_rate: float = 0.05,
    seed: int = 7,
) -> list[float]:
    """Linear regression with mini-batch SGD; return loss after each step (§8.1.3)."""
    rng = np.random.default_rng(seed)
    xs = np.asarray(x_train, dtype=float).ravel()
    ys = np.asarray(y_train, dtype=float).ravel()
    n = len(xs)
    batch_size = max(1, min(int(batch_size), n))
    w = rng.normal(0.0, 0.1)
    b = 0.0
    losses: list[float] = []
    design = np.column_stack([xs, np.ones(n)])
    for _ in range(int(n_steps)):
        idx = rng.choice(n, size=batch_size, replace=batch_size > n)
        x_b = xs[idx]
        y_b = ys[idx]
        pred = w * x_b + b
        error = pred - y_b
        grad_w = float(2.0 * np.mean(error * x_b))
        grad_b = float(2.0 * np.mean(error))
        w -= learning_rate * grad_w
        b -= learning_rate * grad_b
        full_pred = design @ np.array([w, b])
        losses.append(float(np.mean((full_pred - ys) ** 2)))
    return losses
