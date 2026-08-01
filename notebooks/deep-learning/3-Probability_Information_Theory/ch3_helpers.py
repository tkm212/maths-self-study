"""Shared helpers for Deep Learning Ch. 3 (Probability and Information Theory) notebooks."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy import stats

from maths_self_study.probability import (
    bayes_posterior,
    binary_entropy,
    cross_entropy,
    kl_divergence,
    shannon_entropy,
)


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


def plot_binary_entropy_curve() -> go.Figure:
    """Reproduce figure 3.5: Shannon entropy of Bernoulli(p) in nats."""
    p_grid = np.linspace(0.001, 0.999, 200)
    ent = [binary_entropy(float(p)) for p in p_grid]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=p_grid, y=ent, mode="lines", name="H(p)"))
    fig.update_layout(
        title="Binary Shannon entropy (Deep Learning, fig. 3.5)",
        xaxis_title="p = P(x = 1)",
        yaxis_title="Entropy (nats)",
        height=420,
    )
    return fig


def plot_discrete_distribution(
    values: np.ndarray,
    probs: np.ndarray,
    *,
    title: str,
) -> go.Figure:
    fig = go.Figure(go.Bar(x=values, y=probs))
    fig.update_layout(title=title, xaxis_title="x", yaxis_title="P(x)", height=380)
    return fig


def plot_gaussian_pdf(
    mu: float,
    sigma: float,
    *,
    title: str = "Gaussian PDF",
    x_range: tuple[float, float] | None = None,
) -> go.Figure:
    if x_range is None:
        x_range = (mu - 4 * sigma, mu + 4 * sigma)
    xs = np.linspace(x_range[0], x_range[1], 300)
    ys = stats.norm.pdf(xs, loc=mu, scale=sigma)
    fig = go.Figure(go.Scatter(x=xs, y=ys, mode="lines", name=f"N({mu}, {sigma**2})"))
    fig.update_layout(title=title, xaxis_title="x", yaxis_title="p(x)", height=380)
    return fig


def plot_kl_asymmetric(
    xs: np.ndarray,
    p: np.ndarray,
    q: np.ndarray,
) -> go.Figure:
    """Visualise figure 3.6: KL(P||Q) vs KL(Q||P) use different best approximations."""
    fig = make_subplots(rows=1, cols=2, subplot_titles=("D_KL(P || Q)", "D_KL(Q || P)"))
    fig.add_trace(go.Scatter(x=xs, y=p, mode="lines", name="P", line={"color": "royalblue"}), row=1, col=1)
    fig.add_trace(go.Scatter(x=xs, y=q, mode="lines", name="Q", line={"color": "firebrick"}), row=1, col=1)
    fig.add_trace(go.Scatter(x=xs, y=p, mode="lines", name="P", line={"color": "royalblue"}, showlegend=False), row=1, col=2)
    fig.add_trace(go.Scatter(x=xs, y=q, mode="lines", name="Q", line={"color": "firebrick"}, showlegend=False), row=1, col=2)
    fig.update_layout(height=420, title_text="KL divergence is asymmetric (Deep Learning, fig. 3.6)")
    return fig


def summarize_information_measures(p: np.ndarray, q: np.ndarray) -> dict[str, Any]:
    return {
        "H(P)": shannon_entropy(p),
        "H(P, Q)": cross_entropy(p, q),
        "D_KL(P || Q)": kl_divergence(p, q),
        "D_KL(Q || P)": kl_divergence(q, p),
    }


def chain_joint_probability(factors: list[np.ndarray]) -> np.ndarray:
    """
    Build a small chain-structured joint from p(x1) and p(xi | x{i-1}) factors.

    Each factor after the first is a conditional table with shape (n_prev, n_current).
    """
    joint = factors[0][:, None]
    for cond in factors[1:]:
        joint = joint[..., None] * cond
        if joint.ndim > 2:
            joint = np.moveaxis(joint, -2, -1)
            joint = joint.reshape(*joint.shape[:-2], -1)
    return np.squeeze(joint)


__all__ = [
    "bayes_posterior",
    "binary_entropy",
    "chain_joint_probability",
    "cross_entropy",
    "find_project_root",
    "init_paths",
    "kl_divergence",
    "plot_binary_entropy_curve",
    "plot_discrete_distribution",
    "plot_gaussian_pdf",
    "plot_kl_asymmetric",
    "shannon_entropy",
    "summarize_information_measures",
]
