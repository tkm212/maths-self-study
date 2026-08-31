"""Body content for graphical models page."""

from __future__ import annotations

import ch17_helpers as helpers
from ch17_data import load_inputs
from dash import html

from maths_self_study.dashboards.components import graph, metric, text_box


def render_body() -> html.Div:
    inputs = load_inputs()
    try:
        fig_gl, gl_info = helpers.graphical_lasso_demo_figure(p=20, n=120, cv=5, random_state=0)
        fig_sp, sp = helpers.graphical_lasso_edge_count_vs_alpha_figure(p=20, n=120, n_alphas=45, random_state=0)
        fig_pc, pc_info = helpers.partial_correlation_figure(p=12, n=200, random_state=1)
        fig_tm, tm_info = helpers.tmdb_precision_figure(inputs, max_rows=900, cv=5, random_state=0)
        fig_pair, pr = helpers.tmdb_correlation_and_partial_panels_figure(inputs, max_rows=900, cv=5, random_state=0)
        fig_stab, stab_info = helpers.graphical_lasso_stability_figure(
            p=18, n=100, n_bootstrap=45, cv=5, random_state=0
        )
        fig_g, ginfo = helpers.network_sketch_from_precision_figure(
            inputs, max_rows=900, cv=5, edge_weight_quantile=0.5
        )
    except FileNotFoundError as exc:
        return text_box(steps=[str(exc)], title="Data required")

    return html.Div([
        text_box(
            steps=[
                "GraphicalLassoCV selects penalty rho and estimates sparse precision Theta; zeros encode conditional independences (§17.3.1).",
            ],
            title="Graphical lasso: recovering sparse precision",
        ),
        html.Div(
            [
                metric("CV alpha", f"{gl_info['alpha_selected']:.4f}"),
                metric("Frobenius off-diag error", f"{gl_info['frobenius_offdiag_error']:.3f}"),
            ],
            style={"display": "flex", "gap": "12px", "flexWrap": "wrap", "marginBottom": "12px"},
        ),
        graph(fig_gl),
        html.H3(
            "Sparsity of Theta vs penalty",
            style={"marginTop": "20px", "marginBottom": "8px", "color": "#334155"},
        ),
        html.Div(
            [
                metric("CV alpha", f"{sp['alpha_cv']:.4f}"),
                metric("Edges at CV alpha", str(sp["n_edges_at_cv"])),
            ],
            style={"display": "flex", "gap": "12px", "flexWrap": "wrap", "marginBottom": "12px"},
        ),
        graph(fig_sp),
        text_box(
            steps=[
                "Partial correlation between X_j and X_k given all others is derived from Theta_jk and diagonal entries (§17.3.2).",
            ],
            title="Partial correlations (synthetic)",
        ),
        html.Div(
            [metric("Alpha used", f"{pc_info['alpha']:.4f}")],
            style={"display": "flex", "gap": "12px", "flexWrap": "wrap", "marginBottom": "12px"},
        ),
        graph(fig_pc),
        html.H3(
            "TMDB partial correlations",
            style={"marginTop": "20px", "marginBottom": "8px", "color": "#334155"},
        ),
        html.Div(
            [
                metric("Alpha", f"{tm_info['alpha']:.4f}"),
                metric("n", str(tm_info["n"])),
                metric("p", str(tm_info["p"])),
            ],
            style={"display": "flex", "gap": "12px", "flexWrap": "wrap", "marginBottom": "12px"},
        ),
        graph(fig_tm),
        text_box(
            steps=[
                "Sample correlation conflates direct and indirect links; partial correlations from Theta approximate conditioning structure (§17.3.2).",
            ],
            title="Marginal vs partial association (TMDB)",
        ),
        html.Div(
            [
                metric("Alpha", f"{pr['alpha']:.4f}"),
                metric("n", str(pr["n"])),
                metric("p", str(pr["p"])),
            ],
            style={"display": "flex", "gap": "12px", "flexWrap": "wrap", "marginBottom": "12px"},
        ),
        graph(fig_pair),
        text_box(
            steps=[
                "Bootstrap refits at fixed CV alpha show edge selection frequency - stable edges survive resampling (§17.3).",
            ],
            title="Edge stability under bootstrap",
        ),
        html.Div(
            [
                metric("Fixed alpha", f"{stab_info['alpha_fixed']:.4f}"),
                metric("Mean edge freq.", f"{stab_info['mean_edge_freq']:.3f}"),
            ],
            style={"display": "flex", "gap": "12px", "flexWrap": "wrap", "marginBottom": "12px"},
        ),
        graph(fig_stab),
        text_box(
            steps=[
                "Circle layout connects variable pairs with large |Theta_jk| - a visual aid, not a unique embedding (§17.1).",
            ],
            title="Network sketch from precision (TMDB)",
        ),
        html.Div(
            [
                metric("Edges drawn", str(ginfo["n_edges_drawn"])),
                metric("|Theta| threshold", f"{ginfo['threshold']:.4f}"),
                metric("Alpha", f"{ginfo['alpha']:.4f}"),
            ],
            style={"display": "flex", "gap": "12px", "flexWrap": "wrap", "marginBottom": "12px"},
        ),
        graph(fig_g),
    ])
