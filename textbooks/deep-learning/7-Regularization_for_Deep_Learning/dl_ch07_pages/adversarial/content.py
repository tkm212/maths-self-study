"""Body content for the adversarial page."""

from __future__ import annotations

import dl_ch07_helpers as helpers
from dash import html

from maths_self_study.dashboards.components import graph, table
from maths_self_study.dashboards.utils import coerce_float
from maths_self_study.viz.latex import formula_group
from maths_self_study.viz.textbooks.deep_learning.ch7.formulas import ADVERSARIAL_PERTURB


def render_body(epsilon) -> html.Div:
    eps = coerce_float(epsilon, default=helpers.ADV_EPSILON_DEFAULT)
    fig, stats = helpers.plot_adversarial(eps)
    rows = [
        ["Clean x", f"{stats['x']:.4f}"],
        ["Adversarial x", f"{stats['x_adv']:.4f}"],
        ["True y", f"{stats['y']:.4f}"],
        ["Prediction (clean)", f"{stats['pred']:.4f}"],
        ["Prediction (adversarial)", f"{stats['pred_adv']:.4f}"],
        ["MSE (clean)", f"{stats['mse_clean']:.4f}"],
        ["MSE (adversarial)", f"{stats['mse_adv']:.4f}"],
    ]
    return html.Div([
        html.H3("FGSM-style input perturbation"),
        formula_group(
            ("Adversarial perturbation", ADVERSARIAL_PERTURB),
            title="Key formulas (§7.13)",
        ),
        graph(fig),
        table(["Measure", "Value"], rows, caption="Adversarial example summary"),
    ])
