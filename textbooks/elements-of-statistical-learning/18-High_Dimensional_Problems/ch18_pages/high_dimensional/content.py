"""Body content for high-dimensional page."""

from __future__ import annotations

import ch18_helpers as helpers
from ch18_data import load_inputs
from dash import html

from maths_self_study.dashboards.components import graph, metric, text_box


def render_body() -> html.Div:
    inputs = load_inputs()
    try:
        fig_vol, vol = helpers.curse_of_dimensionality_volume_figure(d_max=50)
        fig_reg, reg_info = helpers.highdim_regularization_comparison_figure(
            n=45, p=180, n_nonzero=12, n_cv=5, random_state=0
        )
        fig_path, path_info = helpers.lasso_path_sparsity_figure(n=40, p=120, n_nonzero=8, random_state=1)
        fig_tf, tf = helpers.lasso_true_vs_fitted_figure(n=60, p=120, n_nonzero=10, random_state=0)
        fig_mag, mag = helpers.ridge_vs_lasso_coefficient_magnitude_figure(n=50, p=100, n_nonzero=8, random_state=1)
        fig_scr, scr_info = helpers.marginal_screening_lasso_figure(n=90, p=400, n_nonzero=15, n_cv=5, random_state=0)
        fig_fdr, fdr_info = helpers.fdr_vs_bonferroni_figure(
            n_rep=40, n=100, p=220, n_signal=10, alpha=0.05, random_state=0
        )
        fig_tm, tm_info = helpers.tmdb_with_noise_features_figure(
            inputs, n_noise=200, max_rows=600, n_cv=5, random_state=0
        )
    except FileNotFoundError as exc:
        return text_box(steps=[str(exc)], title="Data required")

    return html.Div([
        text_box(
            steps=[
                "In [0,1]^d, a concentric (1-epsilon)-cube captures (1-epsilon)^d of volume - neighbourhoods empty with fixed N (§18.1).",
            ],
            title="Curse of dimensionality",
        ),
        html.Div(
            [
                metric("epsilon", f"{vol['epsilon']}"),
                metric("(1-epsilon)^30", f"{vol['at_d_30']:.4f}"),
            ],
            style={"display": "flex", "gap": "12px", "flexWrap": "wrap", "marginBottom": "12px"},
        ),
        graph(fig_vol),
        text_box(
            steps=[
                "With n << p and sparse beta, compare RidgeCV, LassoCV, and ElasticNetCV via cross-validated R² (§18.2-18.3).",
            ],
            title="Regularisation comparison (simulation)",
        ),
        html.Div(
            [
                metric("Best method", str(reg_info["best_method"])),
                metric("CV R²", f"{reg_info['best_r2']:.4f}"),
                metric("True nonzeros", str(reg_info["true_nonzeros"])),
            ],
            style={"display": "flex", "gap": "12px", "flexWrap": "wrap", "marginBottom": "12px"},
        ),
        graph(fig_reg),
        html.H3(
            "Lasso path: active set vs penalty",
            style={"marginTop": "20px", "marginBottom": "8px", "color": "#334155"},
        ),
        html.Div(
            [
                metric("Max active", str(path_info["max_active"])),
                metric("n", str(path_info["n"])),
            ],
            style={"display": "flex", "gap": "12px", "flexWrap": "wrap", "marginBottom": "12px"},
        ),
        graph(fig_path),
        text_box(
            steps=[
                "True beta vs lasso beta after StandardScaler - shrinkage and selection pull fitted values toward the origin (§18.2-18.3).",
            ],
            title="Parameter recovery (one draw)",
        ),
        html.Div(
            [metric("corr(beta, beta_hat)", f"{tf['lasso_r_correlation_true_hat']:.3f}")],
            style={"display": "flex", "gap": "12px", "flexWrap": "wrap", "marginBottom": "12px"},
        ),
        graph(fig_tf),
        html.H3(
            "Ridge vs lasso coefficient magnitudes",
            style={"marginTop": "20px", "marginBottom": "8px", "color": "#334155"},
        ),
        html.Div(
            [metric("Lasso nonzeros", str(mag["n_nonzero_lasso"]))],
            style={"display": "flex", "gap": "12px", "flexWrap": "wrap", "marginBottom": "12px"},
        ),
        graph(fig_mag),
        text_box(
            steps=[
                "Screen top k ~ 2n features by marginal |corr(X_j, y)|, then lasso on the submatrix (§18).",
            ],
            title="Marginal screening + lasso",
        ),
        html.Div(
            [
                metric("R² full lasso", f"{scr_info['r2_full']:.4f}"),
                metric("R² screened", f"{scr_info['r2_screened']:.4f}"),
                metric("Screen k", str(scr_info["screen_k"])),
            ],
            style={"display": "flex", "gap": "12px", "flexWrap": "wrap", "marginBottom": "12px"},
        ),
        graph(fig_scr),
        text_box(
            steps=[
                "Bonferroni controls family-wise error; Benjamini-Hochberg controls FDR - different power/guarantee tradeoffs (§18).",
            ],
            title="Multiple testing: Bonferroni vs BH",
        ),
        html.Div(
            [
                metric("Mean TP", str(fdr_info["mean_tp"])),
                metric("Mean FP", str(fdr_info["mean_fp"])),
            ],
            style={"display": "flex", "gap": "12px", "flexWrap": "wrap", "marginBottom": "12px"},
        ),
        graph(fig_fdr),
        text_box(
            steps=[
                "TMDB regression augmented with Gaussian noise columns so p >> n; lasso drives most noise coefs to zero (§18).",
            ],
            title="Real data + noise features",
        ),
        html.Div(
            [
                metric("n", str(tm_info["n"])),
                metric("p", str(tm_info["p"])),
                metric("R² ridge", f"{tm_info['r2_ridge']:.3f}"),
                metric("R² lasso", f"{tm_info['r2_lasso']:.3f}"),
            ],
            style={"display": "flex", "gap": "12px", "flexWrap": "wrap", "marginBottom": "12px"},
        ),
        graph(fig_tm),
    ])
