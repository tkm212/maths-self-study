"""Body content for the Bayes page."""

from __future__ import annotations

import numpy as np
from dash import html

from maths_self_study.dashboards.components import graph, table
from maths_self_study.deep_learning import ch3_helpers as helpers
from maths_self_study.probability import bayes_posterior
from maths_self_study.viz.formulas.ch3 import BAYES_RULE
from maths_self_study.viz.latex import formula


def render_body(prior_d, sens, fpr) -> html.Div:
    prior_d = float(prior_d)
    states = np.array(["disease", "healthy"])
    prior = np.array([prior_d, 1.0 - prior_d])
    likelihood = np.array([float(sens), float(fpr)])
    posterior = bayes_posterior(prior, likelihood)
    fig_med = helpers.plot_bayes_update(states, prior, likelihood, posterior)
    return html.Div([
        html.H3("Rare disease, positive test — base rate dominates"),
        formula(BAYES_RULE, caption="Bayes' rule (§3.5)"),
        graph(fig_med),
        table(
            ["Quantity", "Value"],
            [
                ["P(disease | +)", f"{posterior[0]:.3f}"],
                ["Prior P(disease)", f"{prior_d:.3f}"],
                ["Sensitivity", f"{float(sens):.2f}"],
                ["False positive rate", f"{float(fpr):.2f}"],
            ],
            caption="Rare disease, positive test",
        ),
    ])
