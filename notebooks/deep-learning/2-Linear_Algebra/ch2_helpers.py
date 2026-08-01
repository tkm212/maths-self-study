"""Shared helpers for Deep Learning Ch. 2 (Linear Algebra) notebooks."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import plotly.graph_objects as go


def find_project_root(max_up: int = 12) -> Path:
    p = Path.cwd().resolve()
    for _ in range(max_up):
        if (p / "pyproject.toml").exists():
            return p
        if p.parent == p:
            break
        p = p.parent
    msg = "Could not find project root (pyproject.toml)."
    raise RuntimeError(msg)


def ensure_package_on_path(root: Path) -> None:
    src = root / "src"
    for candidate in (str(src), str(root)):
        if candidate not in sys.path:
            sys.path.insert(0, candidate)


def init_paths() -> tuple[Path, Path]:
    """Return ``(project_root, outputs_dir)`` and put the package on ``sys.path``."""
    root = find_project_root()
    ensure_package_on_path(root)
    outputs = root / "outputs"
    outputs.mkdir(exist_ok=True)
    return root, outputs


def plot_vectors_2d(
    origin: np.ndarray,
    vectors: list[np.ndarray],
    *,
    labels: list[str],
    title: str,
) -> go.Figure:
    fig = go.Figure()
    o = np.asarray(origin, dtype=float).ravel()[:2]
    colors = ["royalblue", "firebrick", "seagreen", "darkorange"]
    for vec, label, color in zip(vectors, labels, colors, strict=False):
        v = np.asarray(vec, dtype=float).ravel()[:2]
        end = o + v
        fig.add_trace(
            go.Scatter(
                x=[o[0], end[0]],
                y=[o[1], end[1]],
                mode="lines+markers",
                name=label,
                line={"color": color, "width": 3},
            )
        )
    fig.update_layout(
        title=title,
        xaxis_title="x1",
        yaxis_title="x2",
        height=420,
        yaxis={"scaleanchor": "x", "scaleratio": 1},
    )
    return fig


def plot_pca_2d(
    data: np.ndarray,
    codes: np.ndarray,
    *,
    title: str,
) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=data[:, 0],
            y=data[:, 1],
            mode="markers",
            name="original",
            marker={"color": "lightgray"},
        )
    )
    fig.add_trace(
        go.Scatter(
            x=codes[:, 0],
            y=codes[:, 1],
            mode="markers",
            name="PCA codes",
            marker={"color": "royalblue"},
        )
    )
    fig.update_layout(title=title, height=420)
    return fig


def plot_explained_variance(variances: np.ndarray) -> go.Figure:
    idx = np.arange(1, len(variances) + 1)
    fig = go.Figure(go.Bar(x=idx, y=variances))
    fig.update_layout(
        title="PCA explained variance",
        xaxis_title="Component",
        yaxis_title="Variance",
        height=380,
    )
    return fig
