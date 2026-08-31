"""Body content for kernel density page."""

from __future__ import annotations

import ch6_helpers as helpers
from ch6_data import load_cls, load_xy
from dash import html

from maths_self_study.dashboards.components import graph, text_box


def render_body(feat, bw) -> html.Div:
    feat = feat or "budget"
    bw = float(bw or 0.3)
    try:
        X_reg, _, _ = load_xy()
        X_cls, y_cls, _ = load_cls()
        fig_kde = helpers.kde_figure(X_reg, feat=feat)
        fig_nb, nb = helpers.naive_bayes_figure(X_cls, y_cls, feat=feat, bw=bw)
    except FileNotFoundError as exc:
        return text_box(steps=[str(exc)], title="Data required")

    priors = ", ".join(f"{c}: {p:.3f}" for c, p in nb["priors"].items())
    return html.Div([
        text_box(
            steps=["KDE: f̂(x) = (1/nλ) Σ K((x−xᵢ)/λ); small λ is spiky, large λ over-smooths."],
            title="Kernel density estimation",
        ),
        graph(fig_kde),
        html.H3(
            "Naive Bayes via class-conditional KDEs",
            style={"marginTop": "20px", "marginBottom": "8px", "color": "#334155"},
        ),
        html.P(f"Class priors: {priors}", style={"color": "#475569", "marginBottom": "8px"}),
        graph(fig_nb),
    ])
