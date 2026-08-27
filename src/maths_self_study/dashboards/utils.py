"""Shared numeric helpers for dashboard callbacks."""

from __future__ import annotations

import logging
import re

import numpy as np

log = logging.getLogger(__name__)

_NUMBER = re.compile(r"[-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?")


def as_matrix(a11: float, a12: float, a21: float, a22: float) -> np.ndarray:
    return np.array([[a11, a12], [a21, a22]], dtype=float)


def coerce_matrix_2x2(
    a11: float | None,
    a12: float | None,
    a21: float | None,
    a22: float | None,
    *,
    fallback: np.ndarray,
) -> np.ndarray:
    """Build a 2x2 matrix from cell inputs, falling back per entry when empty or invalid."""
    default = np.asarray(fallback, dtype=float).reshape(2, 2)
    raw = [a11, a12, a21, a22]
    out: list[float] = []
    for i, value in enumerate(raw):
        if value is None:
            out.append(float(default.ravel()[i]))
            continue
        try:
            out.append(float(value))
        except TypeError, ValueError:
            log.warning("Invalid matrix cell value %r — using fallback", value)
            out.append(float(default.ravel()[i]))
    return np.array(out, dtype=float).reshape(2, 2)


def format_matrix_2x2(matrix: np.ndarray) -> str:
    """Format a 2x2 matrix for editable textarea input."""
    m = np.asarray(matrix, dtype=float).reshape(2, 2)
    cells = [[f"{value:g}" for value in row] for row in m]
    col_widths = [max(len(cells[r][c]) for r in range(2)) for c in range(2)]

    def fmt_row(row: list[str]) -> str:
        inner = "  ".join(val.rjust(col_widths[i]) for i, val in enumerate(row))
        return f"( {inner} )"

    return "\n".join(fmt_row(row) for row in cells)


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
    xs = np.nan_to_num(xs, nan=0.0, posinf=0.0, neginf=0.0)
    xs = np.clip(xs, 0.0, None)
    total = xs.sum()
    if total <= 0:
        return np.full_like(xs, 1.0 / len(xs))
    return xs / total


def coerce_float(value: float | int | None, *, default: float) -> float:
    """Parse a nullable Dash numeric input."""
    if value is None:
        return default
    try:
        return float(value)
    except TypeError, ValueError:
        log.warning("Invalid numeric input %r — using fallback", value)
        return default


def coerce_vector2(
    x0: float | None,
    x1: float | None,
    *,
    fallback: np.ndarray,
) -> np.ndarray:
    fb = np.asarray(fallback, dtype=float).ravel()
    return np.array([coerce_float(x0, default=float(fb[0])), coerce_float(x1, default=float(fb[1]))])


def coerce_floats(values: list[float | None], *, fallback: np.ndarray) -> np.ndarray:
    fb = np.asarray(fallback, dtype=float).ravel()
    return np.array([coerce_float(v, default=float(fb[i])) for i, v in enumerate(values)])


def clamp_prob(value: float | None, *, default: float = 0.5) -> float:
    if value is None:
        return default
    try:
        return float(np.clip(float(value), 0.0, 1.0))
    except TypeError, ValueError:
        return default


def complement_prob(value: float | None, *, default: float = 0.5) -> float:
    return 1.0 - clamp_prob(value, default=default)


def redistribute_simplex(values: list[float | None], index: int, new_value: float | None) -> list[float]:
    """Fix one probability mass; scale the others so the vector sums to 1."""
    n = len(values)
    clamped = [clamp_prob(v, default=1.0 / n) for v in values]
    fixed = clamp_prob(new_value, default=clamped[index])
    remainder = max(0.0, 1.0 - fixed)
    other_indices = [i for i in range(n) if i != index]
    other_sum = sum(clamped[i] for i in other_indices)
    out = clamped.copy()
    out[index] = fixed
    if other_sum > 0:
        for i in other_indices:
            out[i] = remainder * clamped[i] / other_sum
    else:
        share = remainder / len(other_indices)
        for i in other_indices:
            out[i] = share
    return [float(v) for v in out]


def coerce_probs(values: list[float | None], *, fallback: np.ndarray) -> np.ndarray:
    """Build a probability vector from dashboard inputs, falling back per entry when empty."""
    default = np.asarray(fallback, dtype=float).ravel()
    out: list[float] = []
    for i, value in enumerate(values):
        if value is None:
            out.append(float(default[i]))
            continue
        try:
            out.append(max(0.0, float(value)))
        except TypeError, ValueError:
            log.warning("Invalid probability input %r — using fallback", value)
            out.append(float(default[i]))
    return renorm(np.array(out, dtype=float))


def coerce_tensor_3d(
    values: list[int | float | None],
    *,
    fallback: np.ndarray,
    shape: tuple[int, int, int] = (2, 3, 3),
) -> np.ndarray:
    """Build a rank-3 tensor from flat cell inputs (k, then i, then j order)."""
    default = np.asarray(fallback, dtype=float).reshape(shape)
    ni, nj, nk = shape
    out = default.copy()
    expected = ni * nj * nk
    if len(values) != expected:
        log.warning("Expected %d tensor cells, got %d — using fallbacks", expected, len(values))
        return out
    idx = 0
    for k in range(nk):
        for i in range(ni):
            for j in range(nj):
                value = values[idx]
                idx += 1
                if value is None:
                    continue
                try:
                    out[i, j, k] = float(value)
                except TypeError, ValueError:
                    log.warning("Invalid tensor cell value %r — using fallback", value)
    return out
