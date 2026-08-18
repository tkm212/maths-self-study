"""
Numerical computation and optimization utilities (Goodfellow, Bengio & Courville, Ch. 4).

NumPy is the computational backend; these functions mirror notation and workflows
from the Deep Learning textbook.
"""

from __future__ import annotations

import numpy as np


def log_sum_exp(logits: np.ndarray, axis: int | None = None) -> float | np.ndarray:
    """
    Numerically stable log-sum-exp: log Σ exp(z_i) (§4.1).

    Uses the max-subtraction trick to avoid overflow and underflow.
    """
    z = np.asarray(logits, dtype=float)
    z_max = np.max(z, axis=axis, keepdims=True)
    shifted = z - z_max
    out = z_max.squeeze() + np.log(np.sum(np.exp(shifted), axis=axis))
    if axis is None and z.ndim == 1:
        return float(out)
    return out


def softmax_naive(logits: np.ndarray) -> np.ndarray:
    """Softmax without stabilization — can overflow/underflow (§4.1)."""
    z = np.asarray(logits, dtype=float).ravel()
    exp_z = np.exp(z)
    return exp_z / exp_z.sum()


def softmax_stable(logits: np.ndarray) -> np.ndarray:
    """Softmax via max-subtraction: exp(z - max(z)) / Σ exp(z - max(z)) (§4.1)."""
    z = np.asarray(logits, dtype=float).ravel()
    shifted = z - z.max()
    exp_z = np.exp(shifted)
    return exp_z / exp_z.sum()


def condition_number(matrix: np.ndarray) -> float:
    """Condition number kappa(A) = sigma_max / sigma_min for a square matrix (section 4.2)."""
    mat = np.asarray(matrix, dtype=float)
    if mat.ndim != 2 or mat.shape[0] != mat.shape[1]:
        msg = "matrix must be square"
        raise ValueError(msg)
    singular_values = np.linalg.svd(mat, compute_uv=False)
    if singular_values[-1] == 0:
        return float("inf")
    return float(singular_values[0] / singular_values[-1])


def solve_perturbed(
    matrix: np.ndarray,
    rhs: np.ndarray,
    *,
    delta: float = 1e-6,
) -> tuple[np.ndarray, np.ndarray, float]:
    """
    Solve Ax = b and A(x + δx) = b + δb with small perturbation δb (§4.2).

    Returns (x, x_perturbed, relative_error).
    """
    a = np.asarray(matrix, dtype=float)
    b = np.asarray(rhs, dtype=float).ravel()
    x = np.linalg.solve(a, b)
    b_pert = b + delta * np.ones_like(b)
    x_pert = np.linalg.solve(a, b_pert)
    rel_error = float(np.linalg.norm(x_pert - x) / max(np.linalg.norm(x), 1e-12))
    return x, x_pert, rel_error


def gradient_descent_quadratic(
    hessian: np.ndarray,
    gradient_at_start: np.ndarray,
    start: np.ndarray,
    *,
    learning_rate: float,
    n_steps: int,
) -> np.ndarray:
    """
    Gradient descent on f(x) = ½ xᵀHx + gᵀx with fixed learning rate (§4.3).

    Returns path of shape (n_steps + 1, n_features).
    """
    h = np.asarray(hessian, dtype=float)
    g = np.asarray(gradient_at_start, dtype=float).ravel()
    x = np.asarray(start, dtype=float).ravel().copy()
    path = [x.copy()]
    for _ in range(n_steps):
        grad = h @ x + g
        x = x - learning_rate * grad
        path.append(x.copy())
    return np.array(path)


def newton_quadratic(
    hessian: np.ndarray,
    gradient_at_start: np.ndarray,
    start: np.ndarray,
    *,
    n_steps: int,
) -> np.ndarray:
    """
    Newton's method on a quadratic: x ← x - H⁻¹∇f(x) (§4.3, §4.3.1).

    Each step reaches the minimum in one iteration for a true quadratic.
    """
    h = np.asarray(hessian, dtype=float)
    g = np.asarray(gradient_at_start, dtype=float).ravel()
    x = np.asarray(start, dtype=float).ravel().copy()
    path = [x.copy()]
    h_inv = np.linalg.inv(h)
    for _ in range(n_steps):
        grad = h @ x + g
        x = x - h_inv @ grad
        path.append(x.copy())
    return np.array(path)


def quadratic_value(hessian: np.ndarray, linear: np.ndarray, point: np.ndarray) -> float:
    """Evaluate f(x) = ½ xᵀHx + bᵀx at a point."""
    h = np.asarray(hessian, dtype=float)
    b = np.asarray(linear, dtype=float).ravel()
    x = np.asarray(point, dtype=float).ravel()
    return float(0.5 * x @ h @ x + b @ x)


def linear_least_squares(
    design: np.ndarray,
    targets: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Normal-equation least squares: w* = (AᵀA)⁻¹Aᵀb (§4.5).

    Returns (weights, residuals).
    """
    a = np.asarray(design, dtype=float)
    b = np.asarray(targets, dtype=float).ravel()
    if a.ndim != 2:
        msg = "design matrix must be two-dimensional"
        raise ValueError(msg)
    ata = a.T @ a
    atb = a.T @ b
    weights = np.linalg.solve(ata, atb)
    residuals = b - a @ weights
    return weights, residuals
