"""Shared helpers for Deep Learning Ch. 3 (Probability and Information Theory) notebooks."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy import stats

from maths_self_study.math.probability import (
    align_model_to_support,
    bayes_posterior,
    cross_entropy,
    kl_divergence,
    marginalize,
    monty_hall_posterior,
    shannon_entropy,
)
from maths_self_study.viz.graphs import (
    add_vline,
    apply_layout,
    bar_chart,
    contour_chart,
    equal_axes,
    heatmap_chart,
    line_chart,
)
from maths_self_study.viz.marimo import show

# --- Demo fixtures ---

RAIN_TRAFFIC_JOINT = np.array([[0.10, 0.15], [0.25, 0.50]])
RAIN_TRAFFIC_ROW_LABELS = ("dry", "rain")
RAIN_TRAFFIC_COL_LABELS = ("light", "heavy")

INFO_P = np.array([0.40, 0.30, 0.20, 0.10])
INFO_Q = np.array([0.25, 0.25, 0.25, 0.25])
INFO_LABELS = np.array(["a", "b", "c", "d"])

CATEGORICAL_LABELS = np.array(["A", "B", "C", "D"])
CATEGORICAL_PROBS = np.array([0.05, 0.15, 0.30, 0.50])

GAUSSIAN_2D_MEAN = np.array([0.0, 0.0])
GAUSSIAN_2D_COV = np.array([[1.0, 0.6], [0.6, 1.0]])


def plot_binary_entropy_curve() -> go.Figure:
    """Shannon entropy of Bernoulli(p): maximal uncertainty at p = ½."""
    p_grid = np.linspace(0.001, 0.999, 200)
    ent = [-(p * np.log(p) + (1.0 - p) * np.log(1.0 - p)) for p in p_grid]
    fig = line_chart(
        p_grid,
        ent,
        name="H(p)",
        title="Binary entropy — uncertainty peaks at fair coin",
        xaxis_title="p = P(X = 1)",
        yaxis_title="H(X) (nats)",
        height=420,
        hovertemplate="p=%{x:.3f}<br>H=%{y:.3f} nats<extra></extra>",
    )
    add_vline(fig, 0.5, line_dash="dot", line_color="#94a3b8", annotation_text="max at p=½")
    return fig


def plot_discrete_distribution(
    values: np.ndarray,
    probs: np.ndarray,
    *,
    title: str,
    color: str = "#60a5fa",
) -> go.Figure:
    return bar_chart(
        values,
        probs,
        title=title,
        xaxis_title="x",
        yaxis_title="P(x)",
        color=color,
        height=380,
        hovertemplate="x=%{x}<br>P(x)=%{y:.3f}<extra></extra>",
    )


def plot_joint_with_marginals(
    joint: np.ndarray,
    *,
    row_labels: tuple[str, str] = ("A=0", "A=1"),
    col_labels: tuple[str, str] = ("B=0", "B=1"),
    title: str = "Joint, marginals, and conditionals",
) -> go.Figure:
    """2x2 joint heatmap with marginal bars — factorisation made visible."""
    joint = np.asarray(joint, dtype=float)
    p_a = joint.sum(axis=1)
    p_b = joint.sum(axis=0)

    fig = make_subplots(
        rows=2,
        cols=2,
        column_widths=[0.75, 0.25],
        row_heights=[0.75, 0.25],
        specs=[
            [{"type": "heatmap"}, {"type": "bar"}],
            [{"type": "bar"}, {"type": "scatter"}],
        ],
        horizontal_spacing=0.06,
        vertical_spacing=0.08,
    )

    heatmap_chart(
        joint,
        x=list(col_labels),
        y=list(row_labels),
        colorscale="Blues",
        showscale=False,
        text=np.round(joint, 3),
        texttemplate="%{text}",
        hovertemplate="P(A,B)=%{z:.3f}<extra></extra>",
        fig=fig,
        row=1,
        col=1,
    )
    bar_chart(
        list(col_labels),
        p_b,
        color="#93c5fd",
        showlegend=False,
        fig=fig,
        row=1,
        col=2,
    )
    bar_chart(
        list(row_labels),
        p_a,
        color="#60a5fa",
        showlegend=False,
        fig=fig,
        row=2,
        col=1,
    )
    apply_layout(fig, title=title, height=480, showlegend=False)
    fig.update_yaxes(title_text="P(B)", row=1, col=2)
    fig.update_xaxes(title_text="P(A)", row=2, col=1)
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
    fig = line_chart(
        xs,
        ys,
        name=f"N({mu}, {sigma**2:.2f})",
        title=title,
        xaxis_title="x",
        yaxis_title="p(x)",
        height=380,
        fill="tozeroy",
        fillcolor="rgba(37, 99, 235, 0.12)",
        hovertemplate="x=%{x:.2f}<br>p(x)=%{y:.4f}<extra></extra>",
    )
    add_vline(fig, mu, line_dash="dot", line_color="#64748b", annotation_text="μ")
    return fig


def plot_gaussian_2d_contour(
    mean: np.ndarray,
    cov: np.ndarray,
    *,
    title: str = "Bivariate Gaussian — elliptical level sets",
) -> go.Figure:
    mu = np.asarray(mean, dtype=float).ravel()[:2]
    c = np.asarray(cov, dtype=float).reshape(2, 2)
    x, y = np.mgrid[mu[0] - 3 : mu[0] + 3 : 100j, mu[1] - 3 : mu[1] + 3 : 100j]
    pos = np.dstack((x, y))
    rv = stats.multivariate_normal(mu, c)
    z = rv.pdf(pos)

    fig = contour_chart(
        x[:, 0],
        y[0, :],
        z.T,
        colorscale="Blues",
        contours={"coloring": "lines", "showlabels": True},
        title=title,
        height=460,
    )

    eigvals, eigvecs = np.linalg.eigh(c)
    for i, lam in enumerate(eigvals):
        direction = eigvecs[:, i] * np.sqrt(lam) * 2
        fig.add_trace(
            go.Scatter(
                x=[mu[0], mu[0] + direction[0]],
                y=[mu[1], mu[1] + direction[1]],
                mode="lines+markers",
                name=f"s direction {i + 1}",
                line={"width": 3},
                marker={"size": [0, 8]},
            )
        )
    equal_axes(fig)
    return fig


def plot_bayes_update(
    states: np.ndarray,
    prior: np.ndarray,
    likelihood: np.ndarray,
    posterior: np.ndarray,
    *,
    title: str = "Bayes: prior x likelihood ∝ posterior",
) -> go.Figure:
    x = np.arange(len(states))
    fig = go.Figure()
    fig.add_trace(go.Bar(x=x - 0.25, y=prior, width=0.22, name="prior P(x)", marker={"color": "#94a3b8"}))
    fig.add_trace(
        go.Bar(
            x=x, y=likelihood / likelihood.max(), width=0.22, name="likelihood (scaled)", marker={"color": "#fbbf24"}
        )
    )
    fig.add_trace(go.Bar(x=x + 0.25, y=posterior, width=0.22, name="posterior P(x|y)", marker={"color": "#2563eb"}))
    apply_layout(
        fig,
        title=title,
        xaxis={"tickmode": "array", "tickvals": list(x), "ticktext": list(states)},
        yaxis_title="probability",
        barmode="group",
        height=420,
    )
    return fig


def plot_monty_hall(
    door_probs: np.ndarray,
    *,
    chosen: int,
    opened: int,
    title: str = "Monty Hall posterior",
) -> go.Figure:
    labels = [f"door {i}" for i in range(len(door_probs))]
    colors = ["#2563eb" if i != opened else "#94a3b8" for i in range(len(door_probs))]
    fig = go.Figure(
        go.Bar(
            x=labels,
            y=door_probs,
            marker={"color": colors},
            text=[f"{p:.2f}" for p in door_probs],
            textposition="outside",
        )
    )
    fig.add_annotation(
        x=labels[chosen],
        y=door_probs[chosen] + 0.05,
        text="your pick",
        showarrow=False,
        font={"color": "#64748b"},
    )
    apply_layout(fig, title=title, yaxis={"range": [0, 1.05]}, height=400)
    return fig


def plot_kl_asymmetric(
    xs: np.ndarray,
    p: np.ndarray,
    q: np.ndarray,
) -> go.Figure:
    """Two distributions with KL(P‖Q) ≠ KL(Q‖P) — direction matters."""
    q_for_p = align_model_to_support(p, q)
    p_for_q = align_model_to_support(q, p)
    d_pq = kl_divergence(p, q_for_p)
    d_qp = kl_divergence(q, p_for_q)
    fig = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=(f"P vs Q — D_KL(P‖Q) = {d_pq:.3f}", f"Q vs P — D_KL(Q‖P) = {d_qp:.3f}"),
    )
    for col, (a, b, names) in enumerate(
        ((p, q, ("P", "Q")), (q, p, ("Q", "P"))),
        start=1,
    ):
        fig.add_trace(
            go.Scatter(x=xs, y=a, mode="lines", name=names[0], line={"color": "#2563eb", "width": 2}),
            row=1,
            col=col,
        )
        fig.add_trace(
            go.Scatter(x=xs, y=b, mode="lines", name=names[1], line={"color": "#dc2626", "width": 2}),
            row=1,
            col=col,
        )
    apply_layout(fig, height=420, title_text="KL divergence is not symmetric", showlegend=True)
    return fig


def plot_self_information(probs: np.ndarray, *, labels: np.ndarray | None = None) -> go.Figure:
    """I(x) = -log P(x): rare events carry more information."""
    p = np.asarray(probs, dtype=float)
    info = -np.log(np.clip(p, 1e-12, None))
    xs = labels if labels is not None else np.arange(len(p))
    return bar_chart(
        xs,
        info,
        title="Self-information — surprise grows as probability shrinks",
        xaxis_title="outcome",
        yaxis_title="I(x) = -log P(x)",
        color="#7c3aed",
        height=400,
        hovertemplate="I(x)=%{y:.2f} nats<extra></extra>",
    )


def _markov_edge_label(name: str, parent: str, child: str, transition: np.ndarray) -> str:
    """Compact HTML label for a binary conditional factor P(child | parent)."""
    t = np.asarray(transition, dtype=float).reshape(2, 2)
    return (
        f"<b>{name}</b><br>"
        f"{parent}=0: P({child}=0)={t[0, 0]:.2f}, P({child}=1)={t[0, 1]:.2f}<br>"
        f"{parent}=1: P({child}=0)={t[1, 0]:.2f}, P({child}=1)={t[1, 1]:.2f}"
    )


def plot_markov_chain(
    p_x1: np.ndarray,
    p_x2_given_x1: np.ndarray,
    p_x3_given_x2: np.ndarray,
    *,
    title: str = "Markov chain factorisation",
) -> go.Figure:
    """Three-node directed chain with local conditional factors annotated on each edge."""
    p_x1 = np.asarray(p_x1, dtype=float).ravel()[:2]
    t = np.asarray(p_x2_given_x1, dtype=float).reshape(2, 2)
    u = np.asarray(p_x3_given_x2, dtype=float).reshape(2, 2)

    node_x = [0.0, 1.0, 2.0]
    node_y = [0.5, 0.5, 0.5]
    node_labels = ["X₁", "X₂", "X₃"]
    node_radius = 0.09

    fig = go.Figure()
    for start, end in ((0, 1), (1, 2)):
        fig.add_annotation(
            x=node_x[end] - node_radius,
            y=node_y[end],
            ax=node_x[start] + node_radius,
            ay=node_y[start],
            xref="x",
            yref="y",
            axref="x",
            ayref="y",
            showarrow=True,
            arrowhead=3,
            arrowsize=1.2,
            arrowwidth=2.5,
            arrowcolor="#2563eb",
            text="",
        )

    fig.add_trace(
        go.Scatter(
            x=node_x,
            y=node_y,
            mode="markers+text",
            text=node_labels,
            textposition="top center",
            textfont={"size": 15, "color": "#1e3a8a"},
            marker={"size": 54, "color": "#dbeafe", "line": {"width": 2.5, "color": "#2563eb"}},
            hoverinfo="skip",
            showlegend=False,
        )
    )

    fig.add_annotation(
        x=0.0,
        y=0.28,
        text=f"<b>P(X₁)</b><br>P(X₁=0)={p_x1[0]:.2f}<br>P(X₁=1)={p_x1[1]:.2f}",
        showarrow=False,
        font={"size": 11, "color": "#334155"},
        bgcolor="rgba(255,255,255,0.95)",
        bordercolor="#cbd5e1",
        borderwidth=1,
        borderpad=6,
        align="center",
    )
    for x_mid, label in (
        (0.5, _markov_edge_label("P(X₂|X₁)", "X₁", "X₂", t)),
        (1.5, _markov_edge_label("P(X₃|X₂)", "X₂", "X₃", u)),
    ):
        fig.add_annotation(
            x=x_mid,
            y=0.72,
            text=label,
            showarrow=False,
            font={"size": 10, "color": "#334155"},
            bgcolor="#f8fafc",
            bordercolor="#cbd5e1",
            borderwidth=1,
            borderpad=8,
            align="left",
        )

    fig.add_annotation(
        x=1.0,
        y=0.06,
        text="P(x₁, x₂, x₃) = P(x₁) · P(x₂|x₁) · P(x₃|x₂)",
        showarrow=False,
        font={"size": 12, "color": "#64748b"},
    )

    apply_layout(
        fig,
        title=title,
        height=400,
        margin={"l": 30, "r": 30, "t": 60, "b": 40},
        xaxis={"visible": False, "range": [-0.4, 2.4], "fixedrange": True},
        yaxis={"visible": False, "range": [0, 1], "fixedrange": True},
        plot_bgcolor="#ffffff",
    )
    return fig


def summarize_information_measures(p: np.ndarray, q: np.ndarray) -> dict[str, float]:
    p_arr = np.asarray(p, dtype=float)
    q_arr = np.asarray(q, dtype=float)
    q_for_p = align_model_to_support(p_arr, q_arr)
    p_for_q = align_model_to_support(q_arr, p_arr)
    return {
        "H(P)": shannon_entropy(p_arr),
        "H(P, Q)": cross_entropy(p_arr, q_for_p),
        "D_KL(P || Q)": kl_divergence(p_arr, q_for_p),
        "D_KL(Q || P)": kl_divergence(q_arr, p_for_q),
    }


def format_measures(measures: dict[str, float]) -> str:
    return "\n".join(f"{name:16s} = {value:.4f} nats" for name, value in measures.items())


# --- Scenario demos ---


def rain_traffic_conditional(*, heavy_col: int = 1, rain_row: int = 1) -> float:
    p_b = marginalize(RAIN_TRAFFIC_JOINT, axis=0)
    return float(RAIN_TRAFFIC_JOINT[rain_row, heavy_col] / p_b[heavy_col])


def render_rain_traffic(mo: Any) -> Any:
    return mo.vstack([
        show(
            mo,
            plot_joint_with_marginals(
                RAIN_TRAFFIC_JOINT,
                row_labels=RAIN_TRAFFIC_ROW_LABELS,
                col_labels=RAIN_TRAFFIC_COL_LABELS,
                title="Joint table with marginals",
            ),
        ),
        mo.md(f"P(rain | heavy traffic) = `{rain_traffic_conditional()}`"),
    ])


def discrete_moments(
    support: np.ndarray | None = None,
    probs: np.ndarray | None = None,
) -> tuple[float, float, str]:
    xs = np.asarray(support if support is not None else [0, 1, 2, 3], dtype=float)
    ps = np.asarray(probs if probs is not None else [0.1, 0.2, 0.3, 0.4], dtype=float)
    mean = float(np.sum(xs * ps))
    variance = float(np.sum((xs - mean) ** 2 * ps))
    title = f"E[X] = {mean:.1f}, Var(X) = {variance:.2f}"
    return mean, variance, title


def gaussian_demo_figures() -> list[go.Figure]:
    return [
        plot_gaussian_pdf(0.0, 1.0, title="N(0, 1) — standard normal"),
        plot_gaussian_2d_contour(GAUSSIAN_2D_MEAN, GAUSSIAN_2D_COV, title="Bivariate N — elliptical contours"),
    ]


@dataclass(frozen=True)
class BayesScenario:
    states: np.ndarray
    prior: np.ndarray
    likelihood: np.ndarray
    posterior: np.ndarray
    summary: str


def medical_test_scenario() -> BayesScenario:
    states = np.array(["disease", "healthy"])
    prior = np.array([0.01, 0.99])
    likelihood = np.array([0.95, 0.05])
    posterior = bayes_posterior(prior, likelihood)
    return BayesScenario(
        states=states,
        prior=prior,
        likelihood=likelihood,
        posterior=posterior,
        summary=f"P(disease | +) = {posterior[0]:.3f}  —  only ~16% despite 95% sensitivity",
    )


@dataclass(frozen=True)
class MontyHallScenario:
    posterior: np.ndarray
    chosen: int
    opened: int


def monty_hall_scenario(*, chosen: int = 0, opened: int = 1) -> MontyHallScenario:
    return MontyHallScenario(
        posterior=monty_hall_posterior(chosen_door=chosen, opened_door=opened),
        chosen=chosen,
        opened=opened,
    )


@dataclass(frozen=True)
class MarkovChainDemo:
    joint: np.ndarray
    marginal_x3: float
    p_x1: np.ndarray
    p_x2_given_x1: np.ndarray
    p_x3_given_x2: np.ndarray


def markov_chain_demo() -> MarkovChainDemo:
    p_x1 = np.array([0.6, 0.4])
    p_x2_given_x1 = np.array([[0.7, 0.3], [0.2, 0.8]])
    p_x3_given_x2 = np.array([[0.9, 0.1], [0.4, 0.6]])

    joint = np.zeros((2, 2, 2))
    for x1 in (0, 1):
        for x2 in (0, 1):
            for x3 in (0, 1):
                joint[x1, x2, x3] = p_x1[x1] * p_x2_given_x1[x1, x2] * p_x3_given_x2[x2, x3]

    return MarkovChainDemo(
        joint=joint,
        marginal_x3=float(joint[:, :, 1].sum()),
        p_x1=p_x1,
        p_x2_given_x1=p_x2_given_x1,
        p_x3_given_x2=p_x3_given_x2,
    )


# --- Marimo cell renderers (return UI elements — marimo only displays cell outputs) ---


def render_medical_test(mo: Any) -> Any:
    s = medical_test_scenario()
    return mo.vstack([
        show(mo, plot_bayes_update(s.states, s.prior, s.likelihood, s.posterior)),
        mo.md(s.summary),
    ])


def render_monty_hall(mo: Any, *, chosen: int = 0, opened: int = 1) -> Any:
    s = monty_hall_scenario(chosen=chosen, opened=opened)
    return show(mo, plot_monty_hall(s.posterior, chosen=s.chosen, opened=s.opened))


def render_kl_comparison(mo: Any) -> Any:
    measures = summarize_information_measures(INFO_P, INFO_Q)
    return mo.vstack([
        mo.md("```\n" + format_measures(measures) + "\n```"),
        show(mo, plot_kl_asymmetric(np.arange(len(INFO_P)), INFO_P, INFO_Q)),
    ])


def render_discrete_moments(mo: Any) -> Any:
    _, _, chart_title = discrete_moments()
    return show(
        mo,
        plot_discrete_distribution(
            np.asarray([0, 1, 2, 3]),
            np.asarray([0.1, 0.2, 0.3, 0.4]),
            title=chart_title,
        ),
    )


def render_markov_chain(mo: Any) -> Any:
    demo = markov_chain_demo()
    return mo.vstack([
        show(mo, plot_markov_chain(demo.p_x1, demo.p_x2_given_x1, demo.p_x3_given_x2)),
        mo.md(
            f"8-cell joint from 3 local tables — `{demo.joint.shape}` tensor, sum = `{demo.joint.sum()}`  \n"
            f"Marginal P(X₃=1) = `{demo.marginal_x3}`"
        ),
    ])
