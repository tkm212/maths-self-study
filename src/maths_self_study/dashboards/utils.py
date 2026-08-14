"""Shared numeric helpers for dashboard callbacks."""

from __future__ import annotations

import numpy as np


def as_matrix(a11: float, a12: float, a21: float, a22: float) -> np.ndarray:
    return np.array([[a11, a12], [a21, a22]], dtype=float)


def renorm(values: np.ndarray) -> np.ndarray:
    xs = np.asarray(values, dtype=float)
    xs = np.clip(xs, 0.0, None)
    total = xs.sum()
    if total <= 0:
        return np.full_like(xs, 1.0 / len(xs))
    return xs / total
