"""Shared numeric helpers for dashboard callbacks."""

from __future__ import annotations

import logging
import re

import numpy as np

log = logging.getLogger(__name__)

_NUMBER = re.compile(r"[-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?")


def as_matrix(a11: float, a12: float, a21: float, a22: float) -> np.ndarray:
    return np.array([[a11, a12], [a21, a22]], dtype=float)


def format_matrix_2x2(matrix: np.ndarray) -> str:
    """Format a 2x2 matrix for editable textarea input."""
    m = np.asarray(matrix, dtype=float).reshape(2, 2)
    return "\n".join("\t".join(f"{value:g}" for value in row) for row in m)


def parse_matrix_2x2(text: str | None, *, fallback: np.ndarray) -> np.ndarray:
    """Parse a 2x2 matrix from free-form text (rows, tabs, spaces, or commas)."""
    default = np.asarray(fallback, dtype=float).reshape(2, 2)
    if text is None or not str(text).strip():
        return default.copy()

    tokens = _NUMBER.findall(str(text))
    if len(tokens) != 4:
        log.warning("Expected 4 matrix entries, got %d — using previous values", len(tokens))
        return default.copy()

    try:
        return np.array([float(token) for token in tokens], dtype=float).reshape(2, 2)
    except ValueError:
        log.warning("Could not parse matrix text — using previous values")
        return default.copy()


def renorm(values: np.ndarray) -> np.ndarray:
    xs = np.asarray(values, dtype=float)
    xs = np.clip(xs, 0.0, None)
    total = xs.sum()
    if total <= 0:
        return np.full_like(xs, 1.0 / len(xs))
    return xs / total
