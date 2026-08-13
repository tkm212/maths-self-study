"""Deep Learning Ch. 3 — Probability & Information Theory dashboard (Dash).

Run from repo root:
    uv run python notebooks/deep-learning/3-Probability_Information_Theory/dashboard.py
"""

from __future__ import annotations

import numpy as np
from dash import Dash, Input, Output, dcc, html

from maths_self_study.deep_learning import ch3_helpers as helpers
from maths_self_study.probability import bayes_posterior, monty_hall_posterior

PAGES = [
    {"label": "Random variables", "value": "rv"},
    {"label": "Distributions", "value": "dist"},
    {"label": "Bayes' rule", "value": "bayes"},
    {"label": "Information theory", "value": "info"},
    {"label": "Structured models", "value": "markov"},
]


def _num(
    id_: str, label: str, value: float, *, step: float = 0.01, min_: float | None = None, max_: float | None = None
) -> html.Div:
    kwargs: dict = {
        "id": id_,
        "type": "number",
        "value": value,
        "step": step,
        "debounce": True,
        "style": {"width": "100%", "padding": "6px 8px"},
    }
    if min_ is not None:
        kwargs["min"] = min_
    if max_ is not None:
        kwargs["max"] = max_
    return html.Div(
        [
            html.Label(label, style={"fontSize": "0.85rem", "color": "#475569"}),
            dcc.Input(**kwargs),
        ],
        style={"flex": "1", "minWidth": "110px"},
    )


def _slider(id_: str, label: str, min_: float, max_: float, value: float, step: float = 0.01) -> html.Div:
    return html.Div(
        [
            html.Label(label, style={"fontSize": "0.85rem", "color": "#475569"}),
            dcc.Slider(id=id_, min=min_, max=max_, step=step, value=value, tooltip={"placement": "bottom"}),
        ],
        style={"flex": "1", "minWidth": "180px", "padding": "0 8px"},
    )


def _filter_bar(*children: html.Div) -> html.Div:
    return html.Div(
        list(children),
        style={
            "display": "flex",
            "flexWrap": "wrap",
            "gap": "12px",
            "padding": "14px 16px",
            "background": "#f8fafc",
            "border": "1px solid #e2e8f0",
            "borderRadius": "8px",
            "marginBottom": "16px",
        },
    )


def _section(title: str, *children: html.Div) -> html.Div:
    return html.Div(
        [html.Div(title, style={"fontWeight": 600, "width": "100%", "marginBottom": "4px"}), *children],
        style={"display": "flex", "flexWrap": "wrap", "gap": "12px", "width": "100%"},
    )


def _page_shell(title: str, caption: str, filters: html.Div, body_id: str) -> html.Div:
    return html.Div([
        html.H2(title, style={"marginBottom": "4px"}),
        html.P(caption, style={"color": "#64748b", "marginTop": 0}),
        filters,
        html.Div(id=body_id),
    ])


def _pre(text: str) -> html.Pre:
    return html.Pre(text, style={"background": "#f1f5f9", "padding": "12px", "borderRadius": "6px"})


def _metric(label: str, value: str) -> html.Div:
    return html.Div(
        [html.Strong(label), html.Div(value)],
        style={"flex": "1", "padding": "12px", "background": "#f8fafc", "borderRadius": "8px"},
    )


def _renorm(xs: np.ndarray) -> np.ndarray:
    xs = np.asarray(xs, dtype=float)
    xs = np.clip(xs, 0.0, None)
    total = xs.sum()
    if total <= 0:
        return np.full_like(xs, 1.0 / len(xs))
    return xs / total


app = Dash(__name__, title="Deep Learning Ch. 3 — Probability", suppress_callback_exceptions=True)
server = app.server

app.layout = html.Div(
    [
        html.H1("Deep Learning — Chapter 3: Probability & Information Theory"),
        html.P(
            "Interactive demos with live filters for the chapter constants.",
            style={"color": "#64748b"},
        ),
        dcc.Tabs(
            id="page-tabs",
            value="rv",
            children=[dcc.Tab(label=p["label"], value=p["value"]) for p in PAGES],
        ),
        html.Div(id="page-content", style={"marginTop": "18px"}),
        html.Div(
            html.A(
                "Deep Learning Book — Probability and Information Theory",
                href="https://www.deeplearningbook.org/contents/prob.html",
                target="_blank",
            ),
            style={"marginTop": "28px", "fontSize": "0.9rem"},
        ),
    ],
    style={"maxWidth": "1200px", "margin": "0 auto", "padding": "24px 20px", "fontFamily": "system-ui, sans-serif"},
)


@app.callback(Output("page-content", "children"), Input("page-tabs", "value"))
def render_page(page: str):
    j = helpers.RAIN_TRAFFIC_JOINT
    if page == "rv":
        return _page_shell(
            "Probability as bookkeeping",
            "§3.2-3.8 — Joint → marginals (sum out) → conditionals (slice and renormalise).",
            _filter_bar(
                _section(
                    "P(weather, traffic) joint",
                    _num("rv-j00", "dry · light", float(j[0, 0])),
                    _num("rv-j01", "dry · heavy", float(j[0, 1])),
                    _num("rv-j10", "rain · light", float(j[1, 0])),
                    _num("rv-j11", "rain · heavy", float(j[1, 1])),
                ),
                _section(
                    "Discrete moments support",
                    _num("rv-p0", "P(0)", 0.1),
                    _num("rv-p1", "P(1)", 0.2),
                    _num("rv-p2", "P(2)", 0.3),
                    _num("rv-p3", "P(3)", 0.4),
                ),
            ),
            "rv-body",
        )
    if page == "dist":
        c = helpers.CATEGORICAL_PROBS
        cov = helpers.GAUSSIAN_2D_COV
        return _page_shell(
            "The distributions deep learning lives on",
            "§3.9 — Bernoulli (one bit), Categorical (k classes), Gaussian (continuous workhorse).",
            _filter_bar(
                _section(
                    "Categorical probs",
                    _num("dist-c0", "P(A)", float(c[0])),
                    _num("dist-c1", "P(B)", float(c[1])),
                    _num("dist-c2", "P(C)", float(c[2])),
                    _num("dist-c3", "P(D)", float(c[3])),
                ),
                _section(
                    "2D Gaussian covariance",
                    _num("dist-cov11", "Σ₁₁", float(cov[0, 0]), step=0.1),
                    _num("dist-cov12", "Σ₁₂=Σ₂₁", float(cov[0, 1]), step=0.1),
                    _num("dist-cov22", "Σ₂₂", float(cov[1, 1]), step=0.1),
                ),
            ),
            "dist-body",
        )
    if page == "bayes":
        return _page_shell(
            "Bayes' rule — invert conditioning",
            "§3.11 — Prior x likelihood → posterior. Base rates dominate rare-disease tests.",
            _filter_bar(
                _section(
                    "Medical test",
                    _slider("bayes-prior", "P(disease)", 0.001, 0.5, 0.01, 0.001),
                    _slider("bayes-sens", "Sensitivity P(+|disease)", 0.5, 1.0, 0.95),
                    _slider("bayes-fpr", "False positive P(+|healthy)", 0.0, 0.5, 0.05),
                ),
                _section(
                    "Monty Hall",
                    html.Div(
                        [
                            html.Label("Chosen door", style={"fontSize": "0.85rem", "color": "#475569"}),
                            dcc.Dropdown(
                                id="bayes-chosen",
                                options=[{"label": f"Door {i}", "value": i} for i in range(3)],
                                value=0,
                                clearable=False,
                            ),
                        ],
                        style={"flex": "1", "minWidth": "140px"},
                    ),
                    html.Div(
                        [
                            html.Label("Opened door", style={"fontSize": "0.85rem", "color": "#475569"}),
                            dcc.Dropdown(
                                id="bayes-opened",
                                options=[{"label": f"Door {i}", "value": i} for i in range(3)],
                                value=1,
                                clearable=False,
                            ),
                        ],
                        style={"flex": "1", "minWidth": "140px"},
                    ),
                ),
            ),
            "bayes-body",
        )
    if page == "info":
        p, q = helpers.INFO_P, helpers.INFO_Q
        return _page_shell(
            "Information and surprise",
            "§3.13 — I(x) = -log P(x). H(P) averages surprise; H(P,Q) is classification loss; KL is asymmetric.",
            _filter_bar(
                _section(
                    "P (true)",
                    _num("info-p0", "P(a)", float(p[0])),
                    _num("info-p1", "P(b)", float(p[1])),
                    _num("info-p2", "P(c)", float(p[2])),
                    _num("info-p3", "P(d)", float(p[3])),
                ),
                _section(
                    "Q (model)",
                    _num("info-q0", "Q(a)", float(q[0])),
                    _num("info-q1", "Q(b)", float(q[1])),
                    _num("info-q2", "Q(c)", float(q[2])),
                    _num("info-q3", "Q(d)", float(q[3])),
                ),
            ),
            "info-body",
        )
    return _page_shell(
        "Structured models — factor the joint",
        "§3.14 — Each edge is a conditional. RNNs / HMMs / autoregressive LMs are this with neural conditionals.",
        _filter_bar(
            _section(
                "P(X₁)",
                _slider("mk-px1", "P(X₁=0)", 0.0, 1.0, 0.6),
            ),
            _section(
                "P(X₂ | X₁) — P(X₂=0 | ·)",
                _slider("mk-t00", "X₁=0 → X₂=0", 0.0, 1.0, 0.7),
                _slider("mk-t10", "X₁=1 → X₂=0", 0.0, 1.0, 0.2),
            ),
            _section(
                "P(X₃ | X₂) — P(X₃=0 | ·)",
                _slider("mk-u00", "X₂=0 → X₃=0", 0.0, 1.0, 0.9),
                _slider("mk-u10", "X₂=1 → X₃=0", 0.0, 1.0, 0.4),
            ),
        ),
        "markov-body",
    )


@app.callback(
    Output("rv-body", "children"),
    Input("rv-j00", "value"),
    Input("rv-j01", "value"),
    Input("rv-j10", "value"),
    Input("rv-j11", "value"),
    Input("rv-p0", "value"),
    Input("rv-p1", "value"),
    Input("rv-p2", "value"),
    Input("rv-p3", "value"),
)
def update_rv(j00, j01, j10, j11, p0, p1, p2, p3):
    joint = _renorm(np.array([[j00, j01], [j10, j11]], dtype=float))
    fig_joint = helpers.plot_joint_with_marginals(
        joint,
        row_labels=helpers.RAIN_TRAFFIC_ROW_LABELS,
        col_labels=helpers.RAIN_TRAFFIC_COL_LABELS,
        title="Joint table with marginals",
    )
    p_heavy = joint[:, 1].sum()
    cond = float(joint[1, 1] / p_heavy) if p_heavy > 0 else float("nan")

    probs = _renorm(np.array([p0, p1, p2, p3], dtype=float))
    support = np.array([0, 1, 2, 3], dtype=float)
    _, _, title = helpers.discrete_moments(support, probs)
    fig_moments = helpers.plot_discrete_distribution(support, probs, title=title)

    return html.Div([
        dcc.Graph(figure=fig_joint),
        _pre(f"P(rain | heavy traffic) = {cond:.4f}"),
        html.H3("Expectation and variance"),
        html.P("E[X] = centre of mass; Var(X) = spread about the mean.", style={"color": "#64748b"}),
        dcc.Graph(figure=fig_moments),
    ])


@app.callback(
    Output("dist-body", "children"),
    Input("dist-c0", "value"),
    Input("dist-c1", "value"),
    Input("dist-c2", "value"),
    Input("dist-c3", "value"),
    Input("dist-cov11", "value"),
    Input("dist-cov12", "value"),
    Input("dist-cov22", "value"),
)
def update_dist(c0, c1, c2, c3, s11, s12, s22):
    cat = _renorm(np.array([c0, c1, c2, c3], dtype=float))
    cov = np.array([[float(s11), float(s12)], [float(s12), float(s22)]], dtype=float)
    # Keep PSD for contour demo
    eigvals = np.linalg.eigvalsh(cov)
    note = None
    if np.any(eigvals <= 1e-8):
        cov = cov + np.eye(2) * (1e-2 - float(eigvals.min()))
        note = html.P("Covariance nudged to stay positive definite for the contour plot.", style={"color": "#0369a1"})

    fig_entropy = helpers.plot_binary_entropy_curve()
    fig_1d, fig_2d = helpers.gaussian_demo_figures()
    # Rebuild 2d with filterable cov
    fig_2d = helpers.plot_gaussian_2d_contour(
        helpers.GAUSSIAN_2D_MEAN,
        cov,
        title="Bivariate N — elliptical contours",
    )
    fig_cat = helpers.plot_discrete_distribution(
        helpers.CATEGORICAL_LABELS,
        cat,
        title="Softmax target distribution",
    )
    return html.Div([
        html.H3("Bernoulli — maximal uncertainty at p = ½"),
        dcc.Graph(figure=fig_entropy),
        html.H3("Gaussian — elliptical level sets from the covariance"),
        note,
        html.Div(
            [
                dcc.Graph(figure=fig_1d, style={"flex": "1"}),
                dcc.Graph(figure=fig_2d, style={"flex": "1"}),
            ],
            style={"display": "flex", "flexWrap": "wrap", "gap": "8px"},
        ),
        html.H3("Categorical — finite support"),
        dcc.Graph(figure=fig_cat),
    ])


@app.callback(
    Output("bayes-body", "children"),
    Input("bayes-prior", "value"),
    Input("bayes-sens", "value"),
    Input("bayes-fpr", "value"),
    Input("bayes-chosen", "value"),
    Input("bayes-opened", "value"),
)
def update_bayes(prior_d, sens, fpr, chosen, opened):
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
            dcc.Graph(figure=fig_med),
            _pre(summary),
            html.H3("Monty Hall"),
            html.P("Chosen and opened doors must differ.", style={"color": "#b91c1c"}),
        ])

    post = monty_hall_posterior(chosen_door=chosen_i, opened_door=opened_i)
    fig_monty = helpers.plot_monty_hall(post, chosen=chosen_i, opened=opened_i)
    return html.Div([
        html.H3("Rare disease, positive test — base rate dominates"),
        dcc.Graph(figure=fig_med),
        _pre(summary),
        html.H3("Monty Hall — switching wins with probability ⅔"),
        dcc.Graph(figure=fig_monty),
        _pre(f"Posterior over doors: {np.round(post, 3)}"),
    ])


@app.callback(
    Output("info-body", "children"),
    Input("info-p0", "value"),
    Input("info-p1", "value"),
    Input("info-p2", "value"),
    Input("info-p3", "value"),
    Input("info-q0", "value"),
    Input("info-q1", "value"),
    Input("info-q2", "value"),
    Input("info-q3", "value"),
)
def update_info(p0, p1, p2, p3, q0, q1, q2, q3):
    p = _renorm(np.array([p0, p1, p2, p3], dtype=float))
    q = _renorm(np.array([q0, q1, q2, q3], dtype=float))
    fig_self = helpers.plot_self_information(p, labels=helpers.INFO_LABELS)
    measures = helpers.summarize_information_measures(p, q)
    fig_kl = helpers.plot_kl_asymmetric(np.arange(len(p)), p, q)
    return html.Div([
        html.H3("Self-information: -log P(x)"),
        dcc.Graph(figure=fig_self),
        html.H3("Cross-entropy and KL — direction matters"),
        _pre(helpers.format_measures(measures)),
        dcc.Graph(figure=fig_kl),
    ])


@app.callback(
    Output("markov-body", "children"),
    Input("mk-px1", "value"),
    Input("mk-t00", "value"),
    Input("mk-t10", "value"),
    Input("mk-u00", "value"),
    Input("mk-u10", "value"),
)
def update_markov(px1_0, t00, t10, u00, u10):
    p_x1 = np.array([float(px1_0), 1.0 - float(px1_0)])
    p_x2_given_x1 = np.array([
        [float(t00), 1.0 - float(t00)],
        [float(t10), 1.0 - float(t10)],
    ])
    p_x3_given_x2 = np.array([
        [float(u00), 1.0 - float(u00)],
        [float(u10), 1.0 - float(u10)],
    ])
    joint = np.zeros((2, 2, 2))
    for x1 in (0, 1):
        for x2 in (0, 1):
            for x3 in (0, 1):
                joint[x1, x2, x3] = p_x1[x1] * p_x2_given_x1[x1, x2] * p_x3_given_x2[x2, x3]

    fig = helpers.plot_markov_chain(p_x2_given_x1, labels=("X₁", "X₂"))
    return html.Div([
        dcc.Graph(figure=fig),
        html.Div(
            [
                _metric("Joint shape", str(joint.shape)),
                _metric("Joint sum", f"{joint.sum():.4f}"),
                _metric("P(X₃=1)", f"{float(joint[:, :, 1].sum()):.4f}"),
            ],
            style={"display": "flex", "gap": "12px", "flexWrap": "wrap"},
        ),
    ])


if __name__ == "__main__":
    app.run(debug=True)
