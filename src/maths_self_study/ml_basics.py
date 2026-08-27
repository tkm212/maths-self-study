"""
Machine learning basics utilities (Goodfellow, Bengio & Courville, Ch. 5).

NumPy is the computational backend; these functions mirror notation and workflows
from the Deep Learning textbook.
"""

from __future__ import annotations

import numpy as np


def polynomial_features(x: np.ndarray, degree: int) -> np.ndarray:
    """Vandermonde design matrix [1, x, x^2, ..., x^degree]."""
    if degree < 0:
        msg = "degree must be non-negative"
        raise ValueError(msg)
    xs = np.asarray(x, dtype=float).ravel()
    return np.vander(xs, N=degree + 1, increasing=True)


def fit_linear(design: np.ndarray, targets: np.ndarray) -> np.ndarray:
    """Least-squares weights for a design matrix."""
    a = np.asarray(design, dtype=float)
    b = np.asarray(targets, dtype=float).ravel()
    weights, _, _, _ = np.linalg.lstsq(a, b, rcond=None)
    return weights


def predict_linear(design: np.ndarray, weights: np.ndarray) -> np.ndarray:
    return np.asarray(design, dtype=float) @ np.asarray(weights, dtype=float).ravel()


def mean_squared_error(predictions: np.ndarray, targets: np.ndarray) -> float:
    pred = np.asarray(predictions, dtype=float).ravel()
    tgt = np.asarray(targets, dtype=float).ravel()
    return float(np.mean((pred - tgt) ** 2))


def train_test_split(
    x: np.ndarray,
    y: np.ndarray,
    *,
    train_fraction: float = 0.7,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Random train/test split with a fixed seed for reproducible demos."""
    xs = np.asarray(x, dtype=float).ravel()
    ys = np.asarray(y, dtype=float).ravel()
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(xs))
    n_train = max(2, round(train_fraction * len(xs)))
    train_idx = order[:n_train]
    test_idx = order[n_train:]
    return xs[train_idx], ys[train_idx], xs[test_idx], ys[test_idx]


def ridge_fit(design: np.ndarray, targets: np.ndarray, *, l2: float) -> np.ndarray:
    """Ridge regression: w = (X^T X + lambda I)^{-1} X^T y."""
    x = np.asarray(design, dtype=float)
    y = np.asarray(targets, dtype=float).ravel()
    reg = float(l2) * np.eye(x.shape[1])
    return np.linalg.solve(x.T @ x + reg, x.T @ y)


def gaussian_mle(samples: np.ndarray) -> tuple[float, float]:
    """MLE for univariate Gaussian: mean and variance (section 5.5)."""
    data = np.asarray(samples, dtype=float).ravel()
    mean = float(data.mean())
    variance = float(np.mean((data - mean) ** 2))
    return mean, variance


def sgd_linear_regression_path(
    design: np.ndarray,
    targets: np.ndarray,
    *,
    learning_rate: float,
    batch_size: int,
    n_steps: int,
    seed: int = 0,
) -> np.ndarray:
    """SGD path for linear regression; batch_size = n gives full-batch descent."""
    x = np.asarray(design, dtype=float)
    y = np.asarray(targets, dtype=float).ravel()
    n_samples, n_features = x.shape
    batch_size = max(1, min(int(batch_size), n_samples))
    rng = np.random.default_rng(seed)
    weights = np.zeros(n_features)
    path = [weights.copy()]
    for _ in range(n_steps):
        idx = np.arange(n_samples) if batch_size == n_samples else rng.choice(n_samples, size=batch_size, replace=False)
        xb = x[idx]
        yb = y[idx]
        grad = (2.0 / len(idx)) * xb.T @ (xb @ weights - yb)
        weights = weights - learning_rate * grad
        path.append(weights.copy())
    return np.array(path)


def complexity_errors(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
    y_test: np.ndarray,
    *,
    max_degree: int = 12,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Train and test MSE for polynomial degrees 1..max_degree."""
    degrees = np.arange(1, max_degree + 1)
    train_err = np.zeros(len(degrees))
    test_err = np.zeros(len(degrees))
    for i, degree in enumerate(degrees):
        xtr = polynomial_features(x_train, degree)
        xte = polynomial_features(x_test, degree)
        weights = fit_linear(xtr, y_train)
        train_err[i] = mean_squared_error(predict_linear(xtr, weights), y_train)
        test_err[i] = mean_squared_error(predict_linear(xte, weights), y_test)
    return degrees, train_err, test_err
