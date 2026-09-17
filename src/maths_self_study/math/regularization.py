"""
Regularization utilities (Goodfellow, Bengio & Courville, Ch. 7).

NumPy demos for weight decay, early stopping, dropout, and input noise on a
one-hidden-layer MLP fitted to a noisy 1D curve.
"""

from __future__ import annotations

from typing import TypedDict

import numpy as np

from maths_self_study.math.feedforward import ActivationName, activation_deriv, activation_fn


class MlpRegModel(TypedDict):
    W1: np.ndarray
    b1: np.ndarray
    W2: np.ndarray
    b2: np.ndarray
    activation: ActivationName
    x_mean: float
    x_std: float
    y_mean: float
    y_std: float
    train_mse: float
    val_mse: float
    weight_norm: float
    train_mse_history: list[float]
    val_mse_history: list[float]
    weight_norm_history: list[float]
    best_epoch: int
    best_val_mse: float


REGRESSION_HIDDEN = 24
REGRESSION_EPOCHS = 400


def regression_dataset(
    *,
    noise: float = 0.12,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Noisy sine curve with a fixed train/validation split."""
    rng = np.random.default_rng(seed)
    x = np.linspace(-2.0, 2.0, 100)
    y = np.sin(2.0 * x) + 0.35 * np.cos(3.0 * x) + rng.normal(0.0, noise, size=len(x))
    order = rng.permutation(len(x))
    n_train = 70
    train_idx = order[:n_train]
    val_idx = order[n_train:]
    return x[train_idx], y[train_idx], x[val_idx], y[val_idx]


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


def _denorm_predictions(pred_norm: np.ndarray, stats: dict[str, float]) -> np.ndarray:
    return pred_norm * stats["y_std"] + stats["y_mean"]


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


def _weight_norm(w1: np.ndarray, w2: np.ndarray) -> float:
    return float(np.sqrt(np.sum(w1**2) + np.sum(w2**2)))


def _mse_norm(pred_norm: np.ndarray, y_norm: np.ndarray) -> float:
    return float(np.mean((pred_norm.ravel() - y_norm.ravel()) ** 2))


def train_mlp_reg(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    *,
    n_hidden: int = REGRESSION_HIDDEN,
    learning_rate: float = 0.01,
    n_epochs: int = REGRESSION_EPOCHS,
    activation: ActivationName = "tanh",
    l2: float = 0.0,
    dropout: float = 0.0,
    input_noise: float = 0.0,
    seed: int = 1,
) -> MlpRegModel:
    """Train a regularized 1D MLP and record train/validation error each epoch."""
    rng = np.random.default_rng(seed)
    n_hidden = max(2, int(n_hidden))
    x_tr, y_tr, x_va, y_va, stats = _normalize_split(x_train, y_train, x_val, y_val)
    x_feat = x_tr.reshape(-1, 1)
    x_val_feat = x_va.reshape(-1, 1)
    w1 = rng.normal(0.0, 0.5, size=(1, n_hidden))
    b1 = np.zeros(n_hidden)
    w2 = rng.normal(0.0, 0.5, size=(n_hidden, 1))
    b2 = np.zeros(1)
    act = activation_fn(activation)
    d_act = lambda z: activation_deriv(activation, z)
    step = learning_rate / np.sqrt(n_hidden)
    l2 = max(0.0, float(l2))
    dropout = float(np.clip(dropout, 0.0, 0.9))
    input_noise = max(0.0, float(input_noise))

    train_mse_hist: list[float] = []
    val_mse_hist: list[float] = []
    weight_norm_hist: list[float] = []

    for _ in range(int(n_epochs)):
        x_noisy = x_feat
        if input_noise > 0.0:
            x_noisy = x_feat + rng.normal(0.0, input_noise, size=x_feat.shape)

        z1, h, _pred = _forward_mlp(x_noisy, w1, b1, w2, b2, act)
        if dropout > 0.0:
            keep = rng.random(h.shape) > dropout
            h_train = h * keep / (1.0 - dropout)
        else:
            h_train = h

        pred_train = h_train @ w2 + b2
        error = pred_train.ravel() - y_tr
        dz2 = error.reshape(-1, 1)
        dw2 = h_train.T @ dz2 + l2 * w2
        db2 = dz2.sum(axis=0)
        dh = dz2 @ w2.T
        dz1 = dh * d_act(z1)
        dw1 = x_noisy.T @ dz1 + l2 * w1
        db1 = dz1.sum(axis=0)
        w2 -= step * dw2
        b2 -= step * db2
        w1 -= step * dw1
        b1 -= step * db1

        _, _h_eval, pred_eval = _forward_mlp(x_feat, w1, b1, w2, b2, act)
        _, _, pred_val = _forward_mlp(x_val_feat, w1, b1, w2, b2, act)
        train_mse_hist.append(_mse_norm(pred_eval, y_tr))
        val_mse_hist.append(_mse_norm(pred_val, y_va))
        weight_norm_hist.append(_weight_norm(w1, w2))

    _, _, pred_train_final = _forward_mlp(x_feat, w1, b1, w2, b2, act)
    _, _, pred_val_final = _forward_mlp(x_val_feat, w1, b1, w2, b2, act)
    train_pred = _denorm_predictions(pred_train_final.ravel(), stats)
    val_pred = _denorm_predictions(pred_val_final.ravel(), stats)
    ys_tr = np.asarray(y_train, dtype=float).ravel()
    ys_va = np.asarray(y_val, dtype=float).ravel()
    best_epoch = int(np.argmin(val_mse_hist))

    return {
        "W1": w1,
        "b1": b1,
        "W2": w2,
        "b2": b2,
        "activation": activation,
        "x_mean": stats["x_mean"],
        "x_std": stats["x_std"],
        "y_mean": stats["y_mean"],
        "y_std": stats["y_std"],
        "train_mse": float(np.mean((train_pred - ys_tr) ** 2)),
        "val_mse": float(np.mean((val_pred - ys_va) ** 2)),
        "weight_norm": _weight_norm(w1, w2),
        "train_mse_history": train_mse_hist,
        "val_mse_history": val_mse_hist,
        "weight_norm_history": weight_norm_hist,
        "best_epoch": best_epoch,
        "best_val_mse": float(val_mse_hist[best_epoch]),
    }


def predict_mlp_reg(model: MlpRegModel, x_line: np.ndarray) -> np.ndarray:
    """Predict on original-scale x using a model from ``train_mlp_reg``."""
    w1 = model["W1"]
    b1 = model["b1"]
    w2 = model["W2"]
    b2 = model["b2"]
    activation = model["activation"]
    act = activation_fn(activation)
    stats = {
        "x_mean": model["x_mean"],
        "x_std": model["x_std"],
        "y_mean": model["y_mean"],
        "y_std": model["y_std"],
    }
    xs = np.asarray(x_line, dtype=float).ravel()
    x_feat = ((xs - stats["x_mean"]) / stats["x_std"]).reshape(-1, 1)
    _, _, pred = _forward_mlp(x_feat, w1, b1, w2, b2, act)
    return _denorm_predictions(pred.ravel(), stats)


def early_stop_mse(history: list[float], stop_epoch: int) -> float:
    """Validation MSE at a chosen early-stopping epoch (1-indexed)."""
    idx = int(np.clip(stop_epoch - 1, 0, len(history) - 1))
    return float(history[idx])


class SemiSupervisedResult(TypedDict):
    labeled_only_val_mse: float
    semi_supervised_val_mse: float
    n_labeled: int
    n_unlabeled: int


class MultitaskResult(TypedDict):
    shared_val_mse: float
    separate_val_mse: float
    task1_shared: float
    task2_shared: float
    task1_separate: float
    task2_separate: float


class ParameterSharingResult(TypedDict):
    fc_params: int
    conv_params: int
    fc_shift_mse: float
    conv_shift_mse: float
    kernel_sizes: list[int]
    conv_mse_curve: list[float]


class BaggingResult(TypedDict):
    single_val_mse: float
    bagged_val_mse: float
    n_estimators: int
    curve_mses: list[float]


class AdversarialResult(TypedDict):
    x: float
    x_adv: float
    y: float
    pred: float
    pred_adv: float
    mse_clean: float
    mse_adv: float


class TangentDistanceResult(TypedDict):
    euclidean: float
    tangent: float
    x0: float
    x1: float


def semi_supervised_comparison(
    *,
    n_labeled: int = 25,
    consistency_weight: float = 0.5,
    seed: int = 1,
) -> SemiSupervisedResult:
    """Compare labeled-only training vs consistency regularization on unlabeled points (§7.6)."""
    x_tr, y_tr, x_va, y_va = regression_dataset(seed=0)
    n_labeled = int(np.clip(n_labeled, 5, len(x_tr)))
    labeled = train_mlp_reg(
        x_tr[:n_labeled],
        y_tr[:n_labeled],
        x_va,
        y_va,
        seed=seed,
    )
    semi = train_semi_supervised_mlp(
        x_tr,
        y_tr,
        n_labeled=n_labeled,
        consistency_weight=consistency_weight,
        x_val=x_va,
        y_val=y_va,
        seed=seed,
    )
    return {
        "labeled_only_val_mse": labeled["val_mse"],
        "semi_supervised_val_mse": semi["val_mse"],
        "n_labeled": n_labeled,
        "n_unlabeled": len(x_tr) - n_labeled,
    }


def train_semi_supervised_mlp(
    x_train: np.ndarray,
    y_train: np.ndarray,
    *,
    n_labeled: int,
    consistency_weight: float = 0.5,
    x_val: np.ndarray,
    y_val: np.ndarray,
    n_hidden: int = REGRESSION_HIDDEN,
    learning_rate: float = 0.01,
    n_epochs: int = REGRESSION_EPOCHS,
    activation: ActivationName = "tanh",
    seed: int = 1,
) -> MlpRegModel:
    """Train with supervised loss on labeled points and consistency on the rest."""
    rng = np.random.default_rng(seed)
    n_labeled = int(np.clip(n_labeled, 1, len(x_train)))
    x_tr, y_tr, x_va, _y_va, stats = _normalize_split(x_train, y_train, x_val, y_val)
    x_feat = x_tr.reshape(-1, 1)
    x_val_feat = x_va.reshape(-1, 1)
    x_unlabeled = x_feat[n_labeled:]
    y_labeled = y_tr[:n_labeled]
    x_labeled = x_feat[:n_labeled]
    w1 = rng.normal(0.0, 0.5, size=(1, n_hidden))
    b1 = np.zeros(n_hidden)
    w2 = rng.normal(0.0, 0.5, size=(n_hidden, 1))
    b2 = np.zeros(1)
    act = activation_fn(activation)
    d_act = lambda z: activation_deriv(activation, z)
    step = learning_rate / np.sqrt(n_hidden)
    weight = max(0.0, float(consistency_weight))
    noise_sigma = 0.08

    for _ in range(int(n_epochs)):
        z1_l, h_l, pred_l = _forward_mlp(x_labeled, w1, b1, w2, b2, act)
        sup_error = pred_l.ravel() - y_labeled
        dz2 = sup_error.reshape(-1, 1)
        dw2 = h_l.T @ dz2
        db2 = dz2.sum(axis=0)
        dh = dz2 @ w2.T
        dz1 = dh * d_act(z1_l)
        dw1 = x_labeled.T @ dz1
        db1 = dz1.sum(axis=0)

        if weight > 0.0 and len(x_unlabeled) > 0:
            x_noisy = x_unlabeled + rng.normal(0.0, noise_sigma, size=x_unlabeled.shape)
            _, _, pred_u = _forward_mlp(x_unlabeled, w1, b1, w2, b2, act)
            _, _, pred_n = _forward_mlp(x_noisy, w1, b1, w2, b2, act)
            cons_error = pred_u - pred_n
            dz2_u = weight * cons_error
            _, h_u, _ = _forward_mlp(x_unlabeled, w1, b1, w2, b2, act)
            dw2 += h_u.T @ dz2_u
            db2 += dz2_u.sum(axis=0)
            dh_u = dz2_u @ w2.T
            z1_u, _, _ = _forward_mlp(x_unlabeled, w1, b1, w2, b2, act)
            dz1_u = dh_u * d_act(z1_u)
            dw1 += x_unlabeled.T @ dz1_u
            db1 += dz1_u.sum(axis=0)

        w2 -= step * dw2
        b2 -= step * db2
        w1 -= step * dw1
        b1 -= step * db1

    _, _, pred_train = _forward_mlp(x_feat, w1, b1, w2, b2, act)
    _, _, pred_val = _forward_mlp(x_val_feat, w1, b1, w2, b2, act)
    train_pred = _denorm_predictions(pred_train.ravel(), stats)
    val_pred = _denorm_predictions(pred_val.ravel(), stats)
    ys_tr = np.asarray(y_train, dtype=float).ravel()
    ys_va = np.asarray(y_val, dtype=float).ravel()
    return {
        "W1": w1,
        "b1": b1,
        "W2": w2,
        "b2": b2,
        "activation": activation,
        "x_mean": stats["x_mean"],
        "x_std": stats["x_std"],
        "y_mean": stats["y_mean"],
        "y_std": stats["y_std"],
        "train_mse": float(np.mean((train_pred - ys_tr) ** 2)),
        "val_mse": float(np.mean((val_pred - ys_va) ** 2)),
        "weight_norm": _weight_norm(w1, w2),
        "train_mse_history": [],
        "val_mse_history": [],
        "weight_norm_history": [],
        "best_epoch": 0,
        "best_val_mse": float(np.mean((val_pred - ys_va) ** 2)),
    }


def multitask_comparison(*, seed: int = 1) -> MultitaskResult:
    """Shared vs separate hidden layers on two related regression tasks (§7.7)."""
    rng = np.random.default_rng(0)
    x = np.linspace(-2.0, 2.0, 120)
    y1 = np.sin(2.0 * x) + rng.normal(0.0, 0.08, size=len(x))
    y2 = np.cos(3.0 * x) + rng.normal(0.0, 0.08, size=len(x))
    order = rng.permutation(len(x))
    n_train = 80
    train_idx = order[:n_train]
    val_idx = order[n_train:]
    shared = train_multitask_mlp(
        x[train_idx],
        y1[train_idx],
        y2[train_idx],
        x[val_idx],
        y1[val_idx],
        y2[val_idx],
        shared=True,
        seed=seed,
    )
    separate = train_multitask_mlp(
        x[train_idx],
        y1[train_idx],
        y2[train_idx],
        x[val_idx],
        y1[val_idx],
        y2[val_idx],
        shared=False,
        seed=seed,
    )
    return {
        "shared_val_mse": shared["val_mse_total"],
        "separate_val_mse": separate["val_mse_total"],
        "task1_shared": shared["val_mse_task1"],
        "task2_shared": shared["val_mse_task2"],
        "task1_separate": separate["val_mse_task1"],
        "task2_separate": separate["val_mse_task2"],
    }


class MultitaskModel(TypedDict):
    val_mse_task1: float
    val_mse_task2: float
    val_mse_total: float


def train_multitask_mlp(
    x_train: np.ndarray,
    y1_train: np.ndarray,
    y2_train: np.ndarray,
    x_val: np.ndarray,
    y1_val: np.ndarray,
    y2_val: np.ndarray,
    *,
    shared: bool,
    n_hidden: int = 16,
    learning_rate: float = 0.02,
    n_epochs: int = 300,
    activation: ActivationName = "tanh",
    seed: int = 1,
) -> MultitaskModel:
    """Two regression heads with optional shared representation."""
    rng = np.random.default_rng(seed)
    n_hidden = max(4, int(n_hidden))
    x_tr, y1_tr, x_va, _y1_va, stats = _normalize_split(x_train, y1_train, x_val, y1_val)
    _, y2_tr, _, _y2_va, _ = _normalize_split(x_train, y2_train, x_val, y2_val)
    x_feat = x_tr.reshape(-1, 1)
    x_val_feat = x_va.reshape(-1, 1)
    act = activation_fn(activation)
    d_act = lambda z: activation_deriv(activation, z)
    step = learning_rate / np.sqrt(n_hidden)

    w1 = rng.normal(0.0, 0.5, size=(1, n_hidden))
    b1 = np.zeros(n_hidden)
    w1_b = rng.normal(0.0, 0.5, size=(1, n_hidden))
    b1_b = np.zeros(n_hidden)
    w2_a = rng.normal(0.0, 0.5, size=(n_hidden, 1))
    b2_a = np.zeros(1)
    w2_b = rng.normal(0.0, 0.5, size=(n_hidden, 1))
    b2_b = np.zeros(1)

    for _ in range(int(n_epochs)):
        if shared:
            z1 = x_feat @ w1 + b1
            h = act(z1)
            pred1 = h @ w2_a + b2_a
            pred2 = h @ w2_b + b2_b
            err1 = pred1.ravel() - y1_tr
            err2 = pred2.ravel() - y2_tr
            dz2_a = err1.reshape(-1, 1)
            dz2_b = err2.reshape(-1, 1)
            dw2_a = h.T @ dz2_a
            dw2_b = h.T @ dz2_b
            db2_a = dz2_a.sum(axis=0)
            db2_b = dz2_b.sum(axis=0)
            dh = dz2_a @ w2_a.T + dz2_b @ w2_b.T
            dz1 = dh * d_act(z1)
            dw1 = x_feat.T @ dz1
            db1 = dz1.sum(axis=0)
            w2_a -= step * dw2_a
            w2_b -= step * dw2_b
            b2_a -= step * db2_a
            b2_b -= step * db2_b
            w1 -= step * dw1
            b1 -= step * db1
        else:
            z1_a = x_feat @ w1 + b1
            h_a = act(z1_a)
            pred1 = h_a @ w2_a + b2_a
            err1 = pred1.ravel() - y1_tr
            dz2_a = err1.reshape(-1, 1)
            dw2_a = h_a.T @ dz2_a
            db2_a = dz2_a.sum(axis=0)
            dh_a = dz2_a @ w2_a.T
            dz1_a = dh_a * d_act(z1_a)
            dw1 = x_feat.T @ dz1_a
            db1 = dz1_a.sum(axis=0)

            z1_b = x_feat @ w1_b + b1_b
            h_b = act(z1_b)
            pred2 = h_b @ w2_b + b2_b
            err2 = pred2.ravel() - y2_tr
            dz2_b = err2.reshape(-1, 1)
            dw2_b = h_b.T @ dz2_b
            db2_b = dz2_b.sum(axis=0)
            dh_b = dz2_b @ w2_b.T
            dz1_b = dh_b * d_act(z1_b)
            dw1_b = x_feat.T @ dz1_b
            db1_b = dz1_b.sum(axis=0)

            w2_a -= step * dw2_a
            b2_a -= step * db2_a
            w1 -= step * dw1
            b1 -= step * db1
            w2_b -= step * dw2_b
            b2_b -= step * db2_b
            w1_b -= step * dw1_b
            b1_b -= step * db1_b

    if shared:
        z1 = x_val_feat @ w1 + b1
        h = act(z1)
        pred1 = (h @ w2_a + b2_a).ravel()
        pred2 = (h @ w2_b + b2_b).ravel()
    else:
        pred1 = (act(x_val_feat @ w1 + b1) @ w2_a + b2_a).ravel()
        pred2 = (act(x_val_feat @ w1_b + b1_b) @ w2_b + b2_b).ravel()
    pred1 = _denorm_predictions(pred1, stats)
    pred2 = _denorm_predictions(pred2, stats)
    mse1 = float(np.mean((pred1 - y1_val) ** 2))
    mse2 = float(np.mean((pred2 - y2_val) ** 2))
    return {
        "val_mse_task1": mse1,
        "val_mse_task2": mse2,
        "val_mse_total": mse1 + mse2,
    }


def _conv1d_valid(x: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """Valid 1D convolution: x shape (L,), kernel (k,) -> (L-k+1,)."""
    x = np.asarray(x, dtype=float).ravel()
    k = np.asarray(kernel, dtype=float).ravel()
    out_len = len(x) - len(k) + 1
    return np.array([np.dot(x[i : i + len(k)], k) for i in range(out_len)], dtype=float)


def spike_localization_dataset(
    *,
    signal_len: int = 12,
    n_train: int = 160,
    n_test: int = 80,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """1D spike position regression; test spikes are shifted right."""
    rng = np.random.default_rng(seed)
    train_x = np.zeros((n_train, signal_len), dtype=float)
    train_y = np.zeros(n_train, dtype=float)
    for i in range(n_train):
        pos = rng.integers(0, signal_len // 2)
        train_x[i, pos] = 1.0
        train_y[i] = pos / signal_len
    test_x = np.zeros((n_test, signal_len), dtype=float)
    test_y = np.zeros(n_test, dtype=float)
    shift = signal_len // 4
    for i in range(n_test):
        pos = rng.integers(0, max(1, signal_len // 2))
        pos_shift = min(pos + shift, signal_len - 1)
        test_x[i, pos_shift] = 1.0
        test_y[i] = pos_shift / signal_len
    return train_x, train_y, test_x, test_y


def parameter_sharing_comparison(*, kernel_size: int = 3, seed: int = 1) -> ParameterSharingResult:
    """Compare FC vs 1D conv on spike localization with shifted test set (§7.9)."""
    kernel_size = int(np.clip(kernel_size, 2, 5))
    train_x, train_y, test_x, test_y = spike_localization_dataset(seed=0)
    signal_len = train_x.shape[1]
    fc = train_spike_fc(train_x, train_y, seed=seed)
    fc_pred = predict_spike_fc(fc, test_x)
    fc_shift_mse = float(np.mean((fc_pred - test_y) ** 2))
    fc_params = signal_len * fc["n_hidden"] + fc["n_hidden"]
    kernel_sizes = list(range(2, 6))
    conv_mse_curve: list[float] = []
    conv_shift_mse = fc_shift_mse
    conv_params = kernel_size + 2
    for k in kernel_sizes:
        conv = train_spike_conv(train_x, train_y, kernel_size=k, seed=seed)
        conv_pred = predict_spike_conv(conv, test_x)
        mse = float(np.mean((conv_pred - test_y) ** 2))
        conv_mse_curve.append(mse)
        if k == kernel_size:
            conv_shift_mse = mse
            conv_params = conv["kernel_size"] + 2
    return {
        "fc_params": int(fc_params),
        "conv_params": int(conv_params),
        "fc_shift_mse": fc_shift_mse,
        "conv_shift_mse": conv_shift_mse,
        "kernel_sizes": kernel_sizes,
        "conv_mse_curve": conv_mse_curve,
    }


class SpikeFcModel(TypedDict):
    W1: np.ndarray
    b1: np.ndarray
    W2: np.ndarray
    b2: np.ndarray
    n_hidden: int


class SpikeConvModel(TypedDict):
    kernel: np.ndarray
    weight: float
    bias: float
    kernel_size: int


def train_spike_fc(
    x_train: np.ndarray,
    y_train: np.ndarray,
    *,
    n_hidden: int = 24,
    learning_rate: float = 0.03,
    n_epochs: int = 300,
    seed: int = 1,
) -> SpikeFcModel:
    rng = np.random.default_rng(seed)
    x = np.asarray(x_train, dtype=float)
    y = np.asarray(y_train, dtype=float).reshape(-1, 1)
    n_in = x.shape[1]
    w1 = rng.normal(0.0, 0.2, size=(n_in, n_hidden))
    b1 = np.zeros(n_hidden)
    w2 = rng.normal(0.0, 0.2, size=(n_hidden, 1))
    b2 = np.zeros(1)
    step = learning_rate / np.sqrt(n_hidden)
    for _ in range(int(n_epochs)):
        h = np.tanh(x @ w1 + b1)
        pred = h @ w2 + b2
        error = pred - y
        dw2 = h.T @ error
        db2 = error.sum(axis=0)
        dh = error @ w2.T
        dz1 = dh * (1.0 - h**2)
        dw1 = x.T @ dz1
        db1 = dz1.sum(axis=0)
        w2 -= step * dw2
        b2 -= step * db2
        w1 -= step * dw1
        b1 -= step * db1
    return {"W1": w1, "b1": b1, "W2": w2, "b2": b2, "n_hidden": n_hidden}


def predict_spike_fc(model: SpikeFcModel, x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    h = np.tanh(x @ model["W1"] + model["b1"])
    return (h @ model["W2"] + model["b2"]).ravel()


def _spike_conv_feature(row: np.ndarray, kernel: np.ndarray) -> tuple[float, int]:
    conv = _conv1d_valid(row, kernel)
    if conv.size == 0:
        return 0.0, 0
    max_idx = int(np.argmax(conv))
    return float(conv[max_idx]), max_idx


def train_spike_conv(
    x_train: np.ndarray,
    y_train: np.ndarray,
    *,
    kernel_size: int = 3,
    learning_rate: float = 0.05,
    n_epochs: int = 300,
    seed: int = 1,
) -> SpikeConvModel:
    rng = np.random.default_rng(seed)
    x_rows = np.asarray(x_train, dtype=float)
    y = np.asarray(y_train, dtype=float).ravel()
    kernel_size = int(kernel_size)
    kernel = rng.normal(0.0, 0.1, size=kernel_size)
    weight = float(rng.normal(0.0, 0.1))
    bias = 0.0
    step_w = learning_rate
    step_k = learning_rate / kernel_size

    for _ in range(int(n_epochs)):
        for row, target in zip(x_rows, y, strict=True):
            feat, max_idx = _spike_conv_feature(row, kernel)
            pred = weight * feat + bias
            error = pred - target
            weight -= step_w * error * feat
            bias -= step_w * error
            kernel -= step_k * error * weight * row[max_idx : max_idx + kernel_size]

    return {
        "kernel": kernel,
        "weight": weight,
        "bias": bias,
        "kernel_size": kernel_size,
    }


def predict_spike_conv(model: SpikeConvModel, x: np.ndarray) -> np.ndarray:
    x_rows = np.asarray(x, dtype=float)
    kernel = model["kernel"]
    weight = model["weight"]
    bias = model["bias"]
    preds = np.zeros(len(x_rows), dtype=float)
    for i, row in enumerate(x_rows):
        feat, _ = _spike_conv_feature(row, kernel)
        preds[i] = weight * feat + bias
    return preds


def bagging_comparison(
    *,
    n_estimators: int = 5,
    curve_max: int = 15,
    seed: int = 1,
) -> BaggingResult:
    """Bootstrap-averaged MLP vs a single model on validation MSE (§7.11)."""
    x_tr, y_tr, x_va, y_va = regression_dataset(seed=0)
    n_estimators = max(1, int(n_estimators))
    curve_max = max(int(curve_max), n_estimators)
    rng = np.random.default_rng(seed)
    single = train_mlp_reg(x_tr, y_tr, x_va, y_va, seed=seed)
    member_preds: list[np.ndarray] = []
    n = len(x_tr)
    for i in range(curve_max):
        idx = rng.integers(0, n, size=n)
        model = train_mlp_reg(x_tr[idx], y_tr[idx], x_va, y_va, seed=seed + i + 1)
        member_preds.append(predict_mlp_reg(model, x_va))
    curve_mses: list[float] = []
    for m in range(1, curve_max + 1):
        bagged = np.mean(np.stack(member_preds[:m], axis=0), axis=0)
        curve_mses.append(float(np.mean((bagged - y_va) ** 2)))
    return {
        "single_val_mse": single["val_mse"],
        "bagged_val_mse": curve_mses[n_estimators - 1],
        "n_estimators": n_estimators,
        "curve_mses": curve_mses,
    }


def adversarial_regression_example(*, epsilon: float = 0.15, seed: int = 1) -> AdversarialResult:
    """FGSM-style input perturbation on a trained 1D regressor (§7.13)."""
    x_tr, y_tr, x_va, y_va = regression_dataset(seed=0)
    model = train_mlp_reg(x_tr, y_tr, x_va, y_va, seed=seed)
    x0 = float(x_va[len(x_va) // 3])
    y0 = float(y_va[len(y_va) // 3])
    eps = max(0.0, float(epsilon))
    pred0 = float(predict_mlp_reg(model, np.array([x0]))[0])
    grad = _input_gradient_mlp_reg(model, x0, y0)
    x_adv = x0 + eps * np.sign(grad)
    pred_adv = float(predict_mlp_reg(model, np.array([x_adv]))[0])
    return {
        "x": x0,
        "x_adv": x_adv,
        "y": y0,
        "pred": pred0,
        "pred_adv": pred_adv,
        "mse_clean": (pred0 - y0) ** 2,
        "mse_adv": (pred_adv - y0) ** 2,
    }


def _input_gradient_mlp_reg(
    model: MlpRegModel,
    x: float,
    y_true: float,
    delta: float = 1e-4,
) -> float:
    pred_plus = float(predict_mlp_reg(model, np.array([x + delta]))[0])
    pred_minus = float(predict_mlp_reg(model, np.array([x - delta]))[0])
    loss_plus = (pred_plus - y_true) ** 2
    loss_minus = (pred_minus - y_true) ** 2
    return float((loss_plus - loss_minus) / (2.0 * delta))


def tangent_distance_translation(*, shift: float = 0.25, x0: float = 0.4) -> TangentDistanceResult:
    """Euclidean vs tangent distance under horizontal translation on y=sin(2x) (§7.14)."""
    shift = float(shift)
    x1 = x0 + shift
    p0 = np.array([x0, np.sin(2.0 * x0)], dtype=float)
    p1 = np.array([x1, np.sin(2.0 * x1)], dtype=float)
    euclidean = float(np.linalg.norm(p1 - p0))
    tangent = np.array([1.0, 2.0 * np.cos(2.0 * x0)], dtype=float)
    alpha = float(np.dot(p1 - p0, tangent) / np.dot(tangent, tangent))
    projected = p0 + alpha * tangent
    tangent_dist = float(np.linalg.norm(p1 - projected))
    return {
        "euclidean": euclidean,
        "tangent": tangent_dist,
        "x0": x0,
        "x1": x1,
    }
