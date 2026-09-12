"""Body content for kernel density page."""

from __future__ import annotations

import ch6_helpers as helpers
from ch6_data import load_cls, load_xy
from dash import html

from maths_self_study.dashboards.components import graph, text_box
from maths_self_study.viz.latex import formula_group
from maths_self_study.viz.textbooks.elements_of_statistical_learning.ch6.formulas import BAYES_CLASSIFIER_POSTERIOR


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
        formula_group(
            ("Naive Bayes posterior", BAYES_CLASSIFIER_POSTERIOR),
            title="Key formulas (§6.6)",
        ),
        graph(fig_kde),
        html.H3(
            "Naive Bayes via class-conditional KDEs",
            style={"marginTop": "20px", "marginBottom": "8px", "color": "#334155"},
        ),
        html.P(f"Class priors: {priors}", style={"color": "#475569", "marginBottom": "8px"}),
        graph(fig_nb),
    ])
