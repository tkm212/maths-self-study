"""Deep Learning Ch. 2 — Linear Algebra dashboard (Dash).

Run from repo root:
    uv run python notebooks/deep-learning/2-Linear_Algebra/dashboard.py
"""

from __future__ import annotations

import numpy as np
from dash import Dash, Input, Output, dcc, html

from maths_self_study.deep_learning import ch2_helpers as helpers
from maths_self_study.linalg import cosine_similarity, lp_norm, pca_fit, pca_inverse_transform, pca_transform

PAGES = [
    {"label": "Vectors & matrices", "value": "vectors"},
    {"label": "Norms", "value": "norms"},
    {"label": "Eigendecomposition", "value": "eigen"},
    {"label": "SVD", "value": "svd"},
    {"label": "PCA", "value": "pca"},
]


def _num(id_: str, label: str, value: float, *, step: float = 0.1) -> html.Div:
    return html.Div(
        [
            html.Label(label, style={"fontSize": "0.85rem", "color": "#475569"}),
            dcc.Input(
                id=id_,
                type="number",
                value=value,
                step=step,
                debounce=True,
                style={"width": "100%", "padding": "6px 8px"},
            ),
        ],
        style={"flex": "1", "minWidth": "110px"},
    )


def _slider(id_: str, label: str, min_: float, max_: float, value: float, step: float = 0.1) -> html.Div:
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


def _matrix_inputs(prefix: str, defaults: np.ndarray, title: str) -> html.Div:
    return html.Div(
        [
            html.Div(title, style={"fontWeight": 600, "width": "100%", "marginBottom": "4px"}),
            _num(f"{prefix}-a11", "a₁₁", float(defaults[0, 0])),
            _num(f"{prefix}-a12", "a₁₂", float(defaults[0, 1])),
            _num(f"{prefix}-a21", "a₂₁", float(defaults[1, 0])),
            _num(f"{prefix}-a22", "a₂₂", float(defaults[1, 1])),
        ],
        style={"display": "flex", "flexWrap": "wrap", "gap": "12px", "width": "100%"},
    )


def _page_shell(title: str, caption: str, filters: html.Div, body_id: str) -> html.Div:
    return html.Div(
        [
            html.H2(title, style={"marginBottom": "4px"}),
            html.P(caption, style={"color": "#64748b", "marginTop": 0}),
            filters,
            html.Div(id=body_id),
        ]
    )


app = Dash(__name__, title="Deep Learning Ch. 2 — Linear Algebra", suppress_callback_exceptions=True)
server = app.server

app.layout = html.Div(
    [
        html.H1("Deep Learning — Chapter 2: Linear Algebra"),
        html.P(
            "Interactive demos with live filters for the chapter constants.",
            style={"color": "#64748b"},
        ),
        dcc.Tabs(
            id="page-tabs",
            value="vectors",
            children=[dcc.Tab(label=p["label"], value=p["value"]) for p in PAGES],
        ),
        html.Div(id="page-content", style={"marginTop": "18px"}),
        html.Div(
            html.A(
                "Deep Learning Book — Linear Algebra",
                href="https://www.deeplearningbook.org/contents/linear_algebra.html",
                target="_blank",
            ),
            style={"marginTop": "28px", "fontSize": "0.9rem"},
        ),
    ],
    style={"maxWidth": "1200px", "margin": "0 auto", "padding": "24px 20px", "fontFamily": "system-ui, sans-serif"},
)


@app.callback(Output("page-content", "children"), Input("page-tabs", "value"))
def render_page(page: str):
    if page == "vectors":
        return _page_shell(
            "Linear maps as geometry",
            "§2.1–2.2 — A matrix A is a linear map x ↦ Ax. Columns of A are where the basis goes.",
            _filter_bar(
                _matrix_inputs("grid", helpers.GRID_MAP, "Grid map A"),
                _slider("vm-rot", "Rotation (°)", -180, 180, 30, 1),
                _slider("vm-shear", "Shear k", -2.0, 2.0, 0.8, 0.1),
                _slider("vm-range", "Grid range", 0.5, 3.0, 1.5, 0.1),
            ),
            "vectors-body",
        )
    if page == "norms":
        return _page_shell(
            "Norms as geometry",
            "§2.5 — ‖x‖ₚ unit balls: L² circle, L¹ diamond, L∞ square.",
            _filter_bar(
                _num("norm-x1", "x₁", 3.0, step=0.5),
                _num("norm-x2", "x₂", -4.0, step=0.5),
                html.Div(
                    [
                        html.Label("Include L∞", style={"fontSize": "0.85rem", "color": "#475569"}),
                        dcc.Checklist(
                            id="norm-inf",
                            options=[{"label": " L∞", "value": "inf"}],
                            value=["inf"],
                        ),
                    ],
                    style={"flex": "1", "minWidth": "120px"},
                ),
            ),
            "norms-body",
        )
    if page == "eigen":
        return _page_shell(
            "Eigendecomposition — invariant directions",
            "§2.7 — Av = λv. Symmetric A: A = QΛQᵀ.",
            _filter_bar(_matrix_inputs("cov", helpers.COV_2X2, "Matrix A (symmetrised for eigh)")),
            "eigen-body",
        )
    if page == "svd":
        return _page_shell(
            "SVD — every matrix has a geometry",
            "§2.8–2.9 — A = UΣVᵀ. Singular values are axis lengths of the unit ball's image.",
            _filter_bar(
                _matrix_inputs("svd", helpers.SVD_MAP, "Map A"),
                html.Div("Least-squares b", style={"fontWeight": 600, "width": "100%"}),
                _num("svd-b0", "b₀", float(helpers.OVERDETERMINED_B[0]), step=0.5),
                _num("svd-b1", "b₁", float(helpers.OVERDETERMINED_B[1]), step=0.5),
                _num("svd-b2", "b₂", float(helpers.OVERDETERMINED_B[2]), step=0.5),
            ),
            "svd-body",
        )
    return _page_shell(
        "PCA — best low-dimensional view",
        "§2.12 — Orthogonal directions of maximal variance = eigenvectors of the covariance.",
        _filter_bar(
            _num("pca-seed", "RNG seed", 42, step=1),
            _num("pca-n", "Samples", 300, step=50),
            _slider("pca-sx", "Stretch x", 0.5, 5.0, 3.0, 0.1),
            _slider("pca-sy", "Stretch y", 0.1, 2.0, 0.5, 0.1),
        ),
        "pca-body",
    )


def _as_matrix(a11, a12, a21, a22) -> np.ndarray:
    return np.array([[a11, a12], [a21, a22]], dtype=float)


@app.callback(
    Output("vectors-body", "children"),
    Input("grid-a11", "value"),
    Input("grid-a12", "value"),
    Input("grid-a21", "value"),
    Input("grid-a22", "value"),
    Input("vm-rot", "value"),
    Input("vm-shear", "value"),
    Input("vm-range", "value"),
)
def update_vectors(a11, a12, a21, a22, rot, shear, grid_range):
    grid_map = _as_matrix(a11, a12, a21, a22)
    composed = helpers.rotation_2d(rot) @ helpers.shear_2d(shear)
    fig_a = helpers.plot_transformed_grid(grid_map, title="A deforms the plane, but keeps it flat", grid_range=grid_range)
    fig_b = helpers.plot_transformed_grid(
        composed,
        title=f"R({rot}°) ∘ S(k={shear:.1f})",
        grid_range=min(grid_range, 1.5),
    )
    return html.Div(
        [
            html.Div(
                [
                    dcc.Graph(figure=fig_a, style={"flex": "1"}),
                    dcc.Graph(figure=fig_b, style={"flex": "1"}),
                ],
                style={"display": "flex", "flexWrap": "wrap", "gap": "8px"},
            ),
            html.P("Inner product — xᵀy = ‖x‖₂ ‖y‖₂ cos θ"),
            html.Pre(helpers.inner_product_45deg(), style={"background": "#f1f5f9", "padding": "12px"}),
        ]
    )


@app.callback(
    Output("norms-body", "children"),
    Input("norm-x1", "value"),
    Input("norm-x2", "value"),
    Input("norm-inf", "value"),
)
def update_norms(x1, x2, inf_opts):
    p_values: tuple[float, ...] = (1.0, 2.0, np.inf) if inf_opts and "inf" in inf_opts else (1.0, 2.0)
    fig = helpers.plot_lp_unit_balls(p_values=p_values)
    x = np.array([float(x1), float(x2)])
    a = np.array([1.0, 0.0])
    b = np.array([1.0, 1.0])
    summary = (
        f"‖x‖₁ = {lp_norm(x, 1):.4f}   ‖x‖₂ = {lp_norm(x, 2):.4f}   ‖x‖∞ = {lp_norm(x, np.inf):.4f}\n"
        f"cos(e₁, (1,1)) = {cosine_similarity(a, b):.4f}  →  45°"
    )
    return html.Div(
        [
            dcc.Graph(figure=fig),
            html.Pre(summary, style={"background": "#f1f5f9", "padding": "12px"}),
        ]
    )


@app.callback(
    Output("eigen-body", "children"),
    Input("cov-a11", "value"),
    Input("cov-a12", "value"),
    Input("cov-a21", "value"),
    Input("cov-a22", "value"),
)
def update_eigen(a11, a12, a21, a22):
    cov = _as_matrix(a11, a12, a21, a22)
    cov_sym = 0.5 * (cov + cov.T)
    note = None
    if not np.allclose(cov, cov_sym):
        note = html.P("Using the symmetric part (A + Aᵀ)/2 for eigendecomposition.", style={"color": "#0369a1"})
    values, _, fig = helpers.eigendecomposition_demo(cov_sym)
    err = helpers.spectral_reconstruction_error(cov_sym)
    return html.Div(
        [
            note,
            dcc.Graph(figure=fig),
            html.Div(
                [
                    html.Div(
                        [html.Strong("Eigenvalues"), html.Div(str(np.round(values, 3)))],
                        style={"flex": "1", "padding": "12px", "background": "#f8fafc", "borderRadius": "8px"},
                    ),
                    html.Div(
                        [html.Strong("‖A − QΛQᵀ‖"), html.Div(f"{err:.2e}")],
                        style={"flex": "1", "padding": "12px", "background": "#f8fafc", "borderRadius": "8px"},
                    ),
                ],
                style={"display": "flex", "gap": "12px"},
            ),
        ]
    )


@app.callback(
    Output("svd-body", "children"),
    Input("svd-a11", "value"),
    Input("svd-a12", "value"),
    Input("svd-a21", "value"),
    Input("svd-a22", "value"),
    Input("svd-b0", "value"),
    Input("svd-b1", "value"),
    Input("svd-b2", "value"),
)
def update_svd(a11, a12, a21, a22, b0, b1, b2):
    svd_map = _as_matrix(a11, a12, a21, a22)
    fig = helpers.plot_svd_geometry(svd_map, title="Unit circle → ellipse; sᵢ = axis lengths")
    b = np.array([float(b0), float(b1), float(b2)])
    return html.Div(
        [
            dcc.Graph(figure=fig),
            html.Pre(
                helpers.least_squares_summary(helpers.OVERDETERMINED_A, b),
                style={"background": "#f1f5f9", "padding": "12px"},
            ),
        ]
    )


@app.callback(
    Output("pca-body", "children"),
    Input("pca-seed", "value"),
    Input("pca-n", "value"),
    Input("pca-sx", "value"),
    Input("pca-sy", "value"),
)
def update_pca(seed, n_samples, sx, sy):
    n = max(50, int(n_samples or 300))
    rng = np.random.default_rng(int(seed or 42))
    z = rng.normal(size=(n, 2))
    transform = np.array([[float(sx), 1.0], [0.0, float(sy)]])
    data = z @ transform + np.array([2.0, -1.0])
    model = pca_fit(data, n_components=2)
    codes = pca_transform(model, data)
    reconstructed = pca_inverse_transform(model, codes)
    error = float(np.linalg.norm(reconstructed - data))
    demo = helpers.PCADemo(data=data, model=model, codes=codes, reconstruction_error=error)
    figs = helpers.pca_figures(demo)
    return html.Div(
        [
            html.Div(
                [
                    dcc.Graph(figure=figs[0], style={"flex": "1"}),
                    dcc.Graph(figure=figs[1], style={"flex": "1"}),
                ],
                style={"display": "flex", "flexWrap": "wrap", "gap": "8px"},
            ),
            dcc.Graph(figure=figs[2]),
            html.Div(
                [html.Strong("Reconstruction error ‖X̂ − X‖"), html.Div(f"{error:.4f}")],
                style={"padding": "12px", "background": "#f8fafc", "borderRadius": "8px"},
            ),
        ]
    )


if __name__ == "__main__":
    app.run(debug=True)
