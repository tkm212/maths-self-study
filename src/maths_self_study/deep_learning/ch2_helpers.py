"""Shared helpers for Deep Learning Ch. 2 (Linear Algebra) notebooks."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from maths_self_study.linalg import (
    PCAModel,
    cosine_similarity,
    lp_norm,
    moore_penrose_pseudoinverse,
    pca_fit,
    pca_inverse_transform,
    pca_transform,
    symmetric_eigendecomposition,
)

# --- Demo fixtures (notebook cells stay declarative) ---

GRID_MAP = np.array([[1.2, 0.4], [-0.3, 0.9]])
TENSOR_A = np.array([1.0, 2.0])
TENSOR_B = np.array([1.0, -1.0, 0.5])
TENSOR_C = np.array([1.0, 2.0, 3.0])
TENSOR_SHAPE = (2, 3, 3)
TENSOR_DEFAULT = np.einsum("i,j,k->ijk", TENSOR_A, TENSOR_B, TENSOR_C)
COV_2X2 = np.array([[2.0, 0.8], [0.8, 1.0]])
SVD_MAP = np.array([[2.0, 0.5], [0.0, 1.5]])
OVERDETERMINED_A = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
OVERDETERMINED_B = np.array([1.0, 2.0, 3.0])


def display(fig: go.Figure, mo: Any) -> Any:
    """Render a Plotly figure in Marimo. Do not use ``fig.show()`` — it won't appear inline."""
    return mo.ui.plotly(fig)


def show(mo: Any, fig: go.Figure) -> Any:
    return display(fig, mo)


def show_all(mo: Any, *figs: go.Figure) -> Any:
    return mo.vstack([display(fig, mo) for fig in figs])


def _base_layout(**overrides: Any) -> dict[str, Any]:
    layout = {
        "template": "plotly_white",
        "margin": {"l": 60, "r": 30, "t": 60, "b": 50},
        "hovermode": "closest",
    }
    layout.update(overrides)
    return layout


def _equal_axes(fig: go.Figure, **kwargs: Any) -> None:
    fig.update_layout(yaxis={"scaleanchor": "x", "scaleratio": 1, **kwargs.get("yaxis", {})})


def plot_vectors_2d(
    origin: np.ndarray,
    vectors: list[np.ndarray],
    *,
    labels: list[str],
    title: str,
    colors: list[str] | None = None,
) -> go.Figure:
    fig = go.Figure()
    o = np.asarray(origin, dtype=float).ravel()[:2]
    palette = colors or ["#2563eb", "#dc2626", "#059669", "#ea580c", "#7c3aed", "#0891b2"]
    for vec, label, color in zip(vectors, labels, palette, strict=False):
        v = np.asarray(vec, dtype=float).ravel()[:2]
        end = o + v
        fig.add_trace(
            go.Scatter(
                x=[o[0], end[0]],
                y=[o[1], end[1]],
                mode="lines+markers",
                name=label,
                line={"color": color, "width": 3},
                marker={"size": 8},
                hovertemplate=f"{label}<br>x=%{{x:.2f}}, y=%{{y:.2f}}<extra></extra>",
            )
        )
    fig.update_layout(**_base_layout(title=title, xaxis_title="x₁", yaxis_title="x₂", height=440))
    _equal_axes(fig)
    return fig


def suggest_grid_range(
    matrix: np.ndarray,
    *,
    target_extent: float = 2.4,
    min_range: float = 0.6,
    max_range: float = 2.8,
) -> float:
    """Pick an input half-width so the warped grid fills the axes comfortably."""
    m = np.asarray(matrix, dtype=float).reshape(2, 2)
    stretch = float(np.max(np.linalg.svd(m, compute_uv=False)))
    stretch = max(stretch, 1e-6)
    return float(np.clip(target_extent / stretch, min_range, max_range))


def plot_transformed_grid(
    matrix: np.ndarray,
    *,
    title: str = "Linear map as deformed grid",
    grid_range: float | None = None,
    n_lines: int = 9,
) -> go.Figure:
    """Show how a 2x2 matrix acts on a coordinate grid — the geometric view of Ax."""
    m = np.asarray(matrix, dtype=float).reshape(2, 2)
    r = suggest_grid_range(m) if grid_range is None else float(grid_range)
    t = np.linspace(-r, r, n_lines)
    fig = go.Figure()

    for x in t:
        pts = np.column_stack([np.full(n_lines, x), np.linspace(-r, r, n_lines)])
        warped = pts @ m.T
        fig.add_trace(
            go.Scatter(
                x=warped[:, 0],
                y=warped[:, 1],
                mode="lines",
                line={"color": "#93c5fd", "width": 1},
                showlegend=False,
                hoverinfo="skip",
            )
        )
    for y in t:
        pts = np.column_stack([np.linspace(-r, r, n_lines), np.full(n_lines, y)])
        warped = pts @ m.T
        fig.add_trace(
            go.Scatter(
                x=warped[:, 0],
                y=warped[:, 1],
                mode="lines",
                line={"color": "#93c5fd", "width": 1},
                showlegend=False,
                hoverinfo="skip",
            )
        )

    basis = np.eye(2)
    warped_basis = basis @ m.T
    for i, (label, color) in enumerate(zip(["e₁ → Ae₁", "e₂ → Ae₂"], ["#2563eb", "#dc2626"], strict=True)):
        fig.add_trace(
            go.Scatter(
                x=[0, warped_basis[i, 0]],
                y=[0, warped_basis[i, 1]],
                mode="lines+markers",
                name=label,
                line={"color": color, "width": 3},
                marker={"size": 8},
            )
        )

    fig.update_layout(**_base_layout(title=title, height=480))
    _equal_axes(fig)
    return fig


def tensor_product(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> np.ndarray:
    """Rank-3 tensor T[i, j, k] = a[i] b[j] c[k] — outer products stacked by a third axis."""
    av = np.asarray(a, dtype=float).ravel()
    bv = np.asarray(b, dtype=float).ravel()
    cv = np.asarray(c, dtype=float).ravel()
    return np.einsum("i,j,k->ijk", av, bv, cv)


def plot_tensor_slice(
    tensor: np.ndarray,
    *,
    axis: int = 2,
    index: int = 0,
    title: str = "2D slice of a rank-3 tensor",
) -> go.Figure:
    """Heatmap of one 2D face of a 3D tensor — indexing makes rank visible."""
    t = np.asarray(tensor, dtype=float)
    axis = int(axis) % 3
    index = int(np.clip(index, 0, t.shape[axis] - 1))
    if axis == 0:
        slab = t[index, :, :]
        y_label, x_label = "j", "k"
        slice_desc = f"i = {index}"
    elif axis == 1:
        slab = t[:, index, :]
        y_label, x_label = "i", "k"
        slice_desc = f"j = {index}"
    else:
        slab = t[:, :, index]
        y_label, x_label = "i", "j"
        slice_desc = f"k = {index}"

    fig = go.Figure(
        go.Heatmap(
            z=slab,
            colorscale="Blues",
            showscale=True,
            text=np.round(slab, 3),
            texttemplate="%{text}",
            hovertemplate=f"T[{slice_desc}]<br>%{{y}} , %{{x}} = %{{z:.3f}}<extra></extra>",
        )
    )
    fig.update_layout(
        **_base_layout(
            title=f"{title} ({slice_desc})",
            xaxis_title=x_label,
            yaxis_title=y_label,
            height=420,
        )
    )
    return fig


def _tensor_entry_grid(tensor: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Integer (i, j, k) coordinates and flattened values for a rank-3 tensor."""
    t = np.asarray(tensor, dtype=float)
    if t.ndim != 3:
        msg = f"Expected rank-3 tensor, got shape {t.shape}"
        raise ValueError(msg)
    ii, jj, kk = np.indices(t.shape)
    return ii.ravel(), jj.ravel(), kk.ravel(), t.ravel()


def _tensor_slice_mask(
    xs: np.ndarray,
    ys: np.ndarray,
    zs: np.ndarray,
    *,
    axis: int,
    index: int,
) -> np.ndarray:
    if axis == 0:
        return xs == index
    if axis == 1:
        return ys == index
    return zs == index


def plot_tensor_3d(
    tensor: np.ndarray,
    *,
    axis: int | None = None,
    index: int | None = None,
    title: str = "Rank-3 tensor — entry values in 3D",
) -> go.Figure:
    """Interactive 3D grid: marker position is (i, j, k), size and colour encode T[i, j, k]."""
    t = np.asarray(tensor, dtype=float)
    xs, ys, zs, vals = _tensor_entry_grid(t)
    abs_vals = np.abs(vals)
    scale = float(abs_vals.max()) if abs_vals.max() > 0 else 1.0
    sizes = 6.0 + 28.0 * (abs_vals / scale)
    hover = [
        f"T[{int(x)}, {int(y)}, {int(z)}] = {float(v):.3f}"
        for x, y, z, v in zip(xs, ys, zs, vals, strict=True)
    ]
    marker_base: dict[str, Any] = {
        "colorscale": "RdBu",
        "cmid": 0.0,
        "cmin": -scale,
        "cmax": scale,
        "line": {"width": 1, "color": "#334155"},
    }

    fig = go.Figure()
    if axis is not None and index is not None:
        axis = int(axis) % 3
        index = int(np.clip(index, 0, t.shape[axis] - 1))
        on_slice = _tensor_slice_mask(xs, ys, zs, axis=axis, index=index)
        axis_names = ("i", "j", "k")
        slice_label = f"{axis_names[axis]} = {index}"

        fig.add_trace(
            go.Scatter3d(
                x=xs[~on_slice],
                y=ys[~on_slice],
                z=zs[~on_slice],
                mode="markers",
                name="other entries",
                marker={
                    **marker_base,
                    "size": sizes[~on_slice] * 0.55,
                    "color": vals[~on_slice],
                    "opacity": 0.35,
                    "showscale": False,
                },
                hovertext=[hover[i] for i in range(len(hover)) if not on_slice[i]],
                hoverinfo="text",
            )
        )
        fig.add_trace(
            go.Scatter3d(
                x=xs[on_slice],
                y=ys[on_slice],
                z=zs[on_slice],
                mode="markers+text",
                name=slice_label,
                marker={
                    **marker_base,
                    "size": sizes[on_slice],
                    "color": vals[on_slice],
                    "opacity": 0.95,
                    "symbol": "diamond",
                    "showscale": True,
                    "colorbar": {"title": "T[i,j,k]"},
                },
                text=[f"{float(v):.2g}" for v in vals[on_slice]],
                textposition="top center",
                textfont={"size": 10, "color": "#0f172a"},
                hovertext=[hover[i] for i in range(len(hover)) if on_slice[i]],
                hoverinfo="text",
            )
        )
        title = f"{title} — highlight {slice_label}"
    else:
        fig.add_trace(
            go.Scatter3d(
                x=xs,
                y=ys,
                z=zs,
                mode="markers+text",
                marker={
                    **marker_base,
                    "size": sizes,
                    "color": vals,
                    "opacity": 0.9,
                    "showscale": True,
                    "colorbar": {"title": "T[i,j,k]"},
                },
                text=[f"{float(v):.2g}" for v in vals],
                textposition="top center",
                textfont={"size": 9, "color": "#0f172a"},
                hovertext=hover,
                hoverinfo="text",
            )
        )

    ni, nj, nk = t.shape
    fig.update_layout(
        **_base_layout(title=title, height=520, showlegend=True),
        scene={
            "xaxis": {"title": "i", "tickmode": "linear", "dtick": 1, "range": [-0.5, ni - 0.5]},
            "yaxis": {"title": "j", "tickmode": "linear", "dtick": 1, "range": [-0.5, nj - 0.5]},
            "zaxis": {"title": "k", "tickmode": "linear", "dtick": 1, "range": [-0.5, nk - 0.5]},
            "aspectmode": "cube",
            "camera": {"eye": {"x": 1.6, "y": 1.6, "z": 1.2}},
        },
    )
    return fig


def plot_lp_unit_balls(*, p_values: tuple[float, ...] = (1.0, 2.0, np.inf)) -> go.Figure:
    """Unit balls for L¹, L², L∞ — geometry of norm choice."""
    theta = np.linspace(0, 2 * np.pi, 400)
    fig = make_subplots(
        rows=1,
        cols=len(p_values),
        subplot_titles=[f"L{'' if p == np.inf else int(p) if p == p // 1 else p} unit ball" for p in p_values],
    )

    for col, p in enumerate(p_values, start=1):
        if p == np.inf:
            xs = np.array([1, 1, -1, -1, 1], dtype=float)
            ys = np.array([1, -1, -1, 1, 1], dtype=float)
        elif p == 1:
            xs = np.sign(np.cos(theta)) * np.maximum(np.abs(np.cos(theta)), np.abs(np.sin(theta)))
            ys = np.sign(np.sin(theta)) * np.maximum(np.abs(np.cos(theta)), np.abs(np.sin(theta)))
        else:
            xs, ys = np.cos(theta), np.sin(theta)
        fig.add_trace(
            go.Scatter(
                x=xs,
                y=ys,
                mode="lines",
                line={"color": "#2563eb", "width": 2},
                showlegend=False,
            ),
            row=1,
            col=col,
        )
        fig.update_xaxes(scaleanchor=f"y{col if col > 1 else ''}", scaleratio=1, row=1, col=col)

    fig.update_layout(**_base_layout(title="Different norms, different geometries", height=380, showlegend=False))
    return fig


def plot_eigen_geometry(
    matrix: np.ndarray,
    eigenvalues: np.ndarray,
    eigenvectors: np.ndarray,
    *,
    title: str = "Eigenvectors: invariant directions",
) -> go.Figure:
    """Eigenvectors stay on their span; A stretches them by λ."""
    a = np.asarray(matrix, dtype=float).reshape(2, 2)
    fig = plot_transformed_grid(a, title=title, grid_range=1.2, n_lines=7)

    for i in range(min(2, len(eigenvalues))):
        v = eigenvectors[:, i]
        lam = eigenvalues[i]
        color = "#dc2626" if i == 0 else "#059669"
        fig.add_trace(
            go.Scatter(
                x=[0, v[0], lam * v[0]],
                y=[0, v[1], lam * v[1]],
                mode="lines+markers",
                name=f"λ={lam:.2f}, v{i + 1}",
                line={"color": color, "width": 3},
                marker={"size": [0, 8, 10]},
            )
        )
    return fig


def plot_svd_geometry(
    matrix: np.ndarray,
    *,
    title: str = "SVD: circle → ellipse",
) -> go.Figure:
    """Unit circle under A; singular values are axis lengths."""
    a = np.asarray(matrix, dtype=float)
    _, s, vt = np.linalg.svd(a, full_matrices=False)
    theta = np.linspace(0, 2 * np.pi, 200)
    circle = np.column_stack([np.cos(theta), np.sin(theta)])
    ellipse = circle @ a.T

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=circle[:, 0],
            y=circle[:, 1],
            mode="lines",
            name="‖x‖₂ = 1",
            line={"color": "#94a3b8", "width": 2, "dash": "dot"},
        )
    )
    fig.add_trace(
        go.Scatter(
            x=ellipse[:, 0],
            y=ellipse[:, 1],
            mode="lines",
            name="‖Ax‖₂ for ‖x‖₂ = 1",
            line={"color": "#2563eb", "width": 3},
            fill="toself",
            fillcolor="rgba(37, 99, 235, 0.12)",
        )
    )

    for i, sig in enumerate(s[:2]):
        direction = vt[i]
        vec = sig * (a @ direction)
        fig.add_trace(
            go.Scatter(
                x=[0, vec[0]],
                y=[0, vec[1]],
                mode="lines+markers",
                name=f"s{i + 1} = {sig:.2f}",
                line={"width": 2},
            )
        )

    fig.update_layout(**_base_layout(title=title, height=460))
    _equal_axes(fig)
    return fig


def plot_pca_scatter(
    data: np.ndarray,
    *,
    title: str,
    mean: np.ndarray | None = None,
    components: np.ndarray | None = None,
    explained_variance: np.ndarray | None = None,
) -> go.Figure:
    """Original data with optional principal-axis overlay."""
    fig = go.Figure(
        go.Scatter(
            x=data[:, 0],
            y=data[:, 1],
            mode="markers",
            name="samples",
            marker={"color": "#64748b", "size": 5, "opacity": 0.55},
            hovertemplate="x₁=%{x:.2f}<br>x₂=%{y:.2f}<extra></extra>",
        )
    )
    if mean is not None and components is not None and explained_variance is not None:
        mu = np.asarray(mean, dtype=float).ravel()[:2]
        scale = 2.5 * np.sqrt(explained_variance)
        colors = ["#2563eb", "#dc2626"]
        for i, (comp, var, color) in enumerate(zip(components, explained_variance, colors, strict=False)):
            direction = np.asarray(comp, dtype=float).ravel()[:2]
            direction = direction / np.linalg.norm(direction)
            tip = mu + scale[i] * direction
            fig.add_trace(
                go.Scatter(
                    x=[mu[0], tip[0]],
                    y=[mu[1], tip[1]],
                    mode="lines+markers",
                    name=f"PC{i + 1} (λ={var:.2f})",
                    line={"color": color, "width": 3},
                    marker={"size": [0, 9]},
                )
            )
        fig.add_trace(
            go.Scatter(
                x=[mu[0]],
                y=[mu[1]],
                mode="markers",
                name="μ",
                marker={"color": "#0f172a", "size": 8, "symbol": "x"},
            )
        )
    fig.update_layout(**_base_layout(title=title, xaxis_title="feature 1", yaxis_title="feature 2", height=440))
    return fig


def plot_pca_codes(codes: np.ndarray, *, title: str) -> go.Figure:
    fig = go.Figure(
        go.Scatter(
            x=codes[:, 0],
            y=codes[:, 1],
            mode="markers",
            name="PCA codes",
            marker={"color": "#2563eb", "size": 5, "opacity": 0.65},
            hovertemplate="c₁=%{x:.2f}<br>c₂=%{y:.2f}<extra></extra>",
        )
    )
    fig.update_layout(**_base_layout(title=title, xaxis_title="PC 1", yaxis_title="PC 2", height=400))
    return fig


def plot_explained_variance(variances: np.ndarray) -> go.Figure:
    idx = np.arange(1, len(variances) + 1)
    total = float(np.sum(variances))
    cumulative = np.cumsum(variances) / total if total > 0 else variances

    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(
        go.Bar(x=idx, y=variances, name="variance", marker={"color": "#93c5fd"}),
        secondary_y=False,
    )
    fig.add_trace(
        go.Scatter(
            x=idx,
            y=cumulative,
            mode="lines+markers",
            name="cumulative",
            line={"color": "#1d4ed8", "width": 2},
        ),
        secondary_y=True,
    )
    fig.update_layout(**_base_layout(title="Variance captured per component", xaxis_title="Component", height=420))
    fig.update_yaxes(title_text="Variance", secondary_y=False)
    fig.update_yaxes(title_text="Cumulative share", secondary_y=True, range=[0, 1.05])
    return fig


def plot_projection_loss(
    data: np.ndarray,
    direction: np.ndarray,
    *,
    title: str = "1D projection: variance along v",
) -> go.Figure:
    """Show scatter, projection line, and projected points — why PCA maximises variance."""
    mu = data.mean(axis=0)
    v = np.asarray(direction, dtype=float).ravel()[:2]
    v = v / np.linalg.norm(v)
    centered = data - mu
    coords = centered @ v

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=data[:, 0],
            y=data[:, 1],
            mode="markers",
            name="data",
            marker={"color": "#64748b", "size": 5, "opacity": 0.5},
        )
    )
    line_t = np.linspace(coords.min() - 0.5, coords.max() + 0.5, 2)
    line_pts = mu + np.outer(line_t, v)
    fig.add_trace(
        go.Scatter(
            x=line_pts[:, 0],
            y=line_pts[:, 1],
            mode="lines",
            name="projection axis",
            line={"color": "#2563eb", "width": 2, "dash": "dash"},
        )
    )
    projected = mu + np.outer(coords, v)
    for i in range(0, len(data), 15):
        fig.add_trace(
            go.Scatter(
                x=[data[i, 0], projected[i, 0]],
                y=[data[i, 1], projected[i, 1]],
                mode="lines",
                line={"color": "#cbd5e1", "width": 1},
                showlegend=False,
                hoverinfo="skip",
            )
        )
    fig.add_trace(
        go.Scatter(
            x=projected[:, 0],
            y=projected[:, 1],
            mode="markers",
            name="projections",
            marker={"color": "#2563eb", "size": 4, "opacity": 0.7},
        )
    )
    fig.update_layout(**_base_layout(title=title, height=440))
    return fig


# --- Small linear-algebra demos ---


def rotation_2d(degrees: float) -> np.ndarray:
    theta = np.deg2rad(degrees)
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c, -s], [s, c]])


def shear_2d(k: float = 0.8) -> np.ndarray:
    return np.array([[1.0, k], [0.0, 1.0]])


def inner_product_45deg() -> str:
    x = np.array([1.0, 0.0])
    y = np.array([1.0, 1.0]) / np.sqrt(2)
    return f"x · y = {float(x @ y):.4f}  (cos 45° = 1/√2)"


def norm_summary() -> str:
    x = np.array([3.0, -4.0])
    a = np.array([1.0, 0.0])
    b = np.array([1.0, 1.0])
    return (
        f"‖x‖₁ = {lp_norm(x, 1):.0f}   ‖x‖₂ = {lp_norm(x, 2):.0f}   (same point, different 'size')\n"
        f"cos(e₁, (1,1)) = {cosine_similarity(a, b):.4f}  →  45°"
    )


def spectral_reconstruction_error(matrix: np.ndarray) -> float:
    values, vectors = np.linalg.eigh(matrix)
    return float(np.linalg.norm(vectors @ np.diag(values) @ vectors.T - matrix))


def least_squares_summary(
    a: np.ndarray = OVERDETERMINED_A,
    b: np.ndarray = OVERDETERMINED_B,
) -> str:
    x = moore_penrose_pseudoinverse(a) @ b
    residual = float(np.linalg.norm(a @ x - b))
    sigmas = np.round(np.linalg.svd(a, compute_uv=False), 4)
    return f"Least-squares x: {np.round(x, 4)}\nResidual ‖Ax - b‖₂: {residual}\nSingular values: {sigmas}"


@dataclass(frozen=True)
class PCADemo:
    data: np.ndarray
    model: PCAModel
    codes: np.ndarray
    reconstruction_error: float


def pca_demo(*, seed: int = 42) -> PCADemo:
    rng = np.random.default_rng(seed)
    z = rng.normal(size=(300, 2))
    transform = np.array([[3.0, 1.0], [0.0, 0.5]])
    data = z @ transform + np.array([2.0, -1.0])
    model = pca_fit(data, n_components=2)
    codes = pca_transform(model, data)
    reconstructed = pca_inverse_transform(model, codes)
    error = float(np.linalg.norm(reconstructed - data))
    return PCADemo(data=data, model=model, codes=codes, reconstruction_error=error)


def pca_figures(demo: PCADemo) -> list[go.Figure]:
    model = demo.model
    return [
        plot_pca_scatter(
            demo.data,
            title="Data + principal axes (length ∝ √λ)",
            mean=model.mean,
            components=model.components,
            explained_variance=model.explained_variance,
        ),
        plot_projection_loss(demo.data, model.components[0], title="Project onto 1st PC"),
        plot_explained_variance(model.explained_variance),
    ]


def eigendecomposition_demo(
    matrix: np.ndarray = COV_2X2,
) -> tuple[np.ndarray, np.ndarray, go.Figure]:
    values, vectors = symmetric_eigendecomposition(matrix)
    fig = plot_eigen_geometry(matrix, values, vectors, title="Eigenvectors span the ellipse axes")
    return values, vectors, fig


# --- Marimo cell renderers (return UI elements — marimo only displays cell outputs) ---


def render_eigendecomposition(mo: Any) -> Any:
    values, _, fig = eigendecomposition_demo()
    return mo.vstack([
        show(mo, fig),
        mo.md(f"Eigenvalues (variance along each axis): `{np.round(values, 3)}`"),
    ])


def render_pca(mo: Any) -> Any:
    demo = pca_demo()
    return mo.vstack([
        show_all(mo, *pca_figures(demo)),
        mo.md(f"Reconstruction error: `{demo.reconstruction_error}`"),
    ])
