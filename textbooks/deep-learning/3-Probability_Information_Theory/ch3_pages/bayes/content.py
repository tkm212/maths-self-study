"""Body content for the Bayes page."""

from __future__ import annotations

import numpy as np
from dash import html

from maths_self_study.dashboards.components import graph, preformatted
from maths_self_study.deep_learning import ch3_helpers as helpers
from maths_self_study.probability import bayes_posterior, monty_hall_posterior


def render_body(prior_d, sens, fpr, chosen, opened) -> html.Div:
    prior_d = float(prior_d)
    states = np.array(["disease", "healthy"])
    prior = np.array([prior_d, 1.0 - prior_d])
    likelihood = np.array([float(sens), float(fpr)])
    posterior = bayes_posterior(prior, likelihood)
    summary = f"P(disease | +) = {posterior[0]:.3f}  —  prior was {prior_d:.3f}, sensitivity {sens:.2f}"
    fig_med = helpers.plot_bayes_update(states, prior, likelihood, posterior)

    chosen_i = int(chosen)
    opened_i = int(opened)
    if chosen_i == opened_i:
        return html.Div([
            html.H3("Rare disease, positive test"),
            graph(fig_med),
            preformatted(summary),
            html.H3("Monty Hall"),
            html.P("Chosen and opened doors must differ.", style={"color": "#b91c1c"}),
        ])

    post = monty_hall_posterior(chosen_door=chosen_i, opened_door=opened_i)
    fig_monty = helpers.plot_monty_hall(post, chosen=chosen_i, opened=opened_i)
    return html.Div([
        html.H3("Rare disease, positive test — base rate dominates"),
        graph(fig_med),
        preformatted(summary),
        html.H3("Monty Hall — switching wins with probability ⅔"),
        graph(fig_monty),
        preformatted(f"Posterior over doors: {np.round(post, 3)}"),
    ])
