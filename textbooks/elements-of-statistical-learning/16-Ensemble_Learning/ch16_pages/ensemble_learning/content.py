"""Body content for ensemble learning page."""

from __future__ import annotations

import ch16_helpers as helpers
from ch16_data import load_regression_xy, load_xy
from dash import html

from maths_self_study.dashboards.components import graph, metric, text_box


def render_body() -> html.Div:
    X, y, _ = load_xy()
    X_reg, y_reg, _ = load_regression_xy()
    try:
        fig_tbf, tbf = helpers.tree_bagging_rf_figure(X, y, n_cv=5)
        fig_oob, oob_info = helpers.oob_error_vs_n_trees_figure(X, y, random_state=0)
        fig_ens, ens_summary = helpers.ensemble_comparison_figure(X, y, n_cv=5)
        fig_w, w_info = helpers.stacking_meta_learned_weights_figure(X, y, n_cv=5)
        fig_meta, meta_summary = helpers.meta_learner_sweep_figure(X, y, n_cv=5)
        fig_div, div_info = helpers.base_error_correlation_figure(X, y, n_cv=5)
        fig_reg, reg_stack = helpers.regression_stacking_figure(X_reg, y_reg, n_cv=5)
    except FileNotFoundError as exc:
        return text_box(steps=[str(exc)], title="Data required")

    return html.Div([
        text_box(
            steps=[
                "A single full-depth tree is high variance; bagging averages bootstrap trees and RF adds feature randomisation at splits (§16.1, Ch. 8, 15).",
            ],
            title="From one tree to bagging to random forest",
        ),
        html.Div(
            [
                metric("Best method", str(tbf["best"])),
                metric("CV accuracy", f"{tbf['best_acc']:.3%}"),
            ],
            style={"display": "flex", "gap": "12px", "flexWrap": "wrap", "marginBottom": "12px"},
        ),
        graph(fig_tbf),
        html.H3(
            "OOB error vs number of trees",
            style={"marginTop": "20px", "marginBottom": "8px", "color": "#334155"},
        ),
        html.Div(
            [metric("Final 1 - OOB", f"{oob_info['final_oob_1minus']:.4f}")],
            style={"display": "flex", "gap": "12px", "flexWrap": "wrap", "marginBottom": "12px"},
        ),
        graph(fig_oob),
        text_box(
            steps=[
                "Compare soft voting (uniform blend of base probabilities) with stacking (logistic meta-learner on OOF outputs) (§16.2).",
            ],
            title="Stacking vs soft voting vs base learners",
        ),
        html.Div(
            [
                metric("Best method", str(ens_summary["best_method"])),
                metric("CV accuracy", f"{ens_summary['best_cv_accuracy']:.3%}"),
            ],
            style={"display": "flex", "gap": "12px", "flexWrap": "wrap", "marginBottom": "12px"},
        ),
        graph(fig_ens),
        text_box(
            steps=[
                "Meta-learner coefficients on log-odds scale show how much each base model's OOF probability is relied on (§16.2).",
            ],
            title="Meta-learner learned weights",
        ),
        html.Div(
            [
                metric("Intercept", f"{w_info['intercept']:.4f}"),
            ]
            + [metric(k, f"{v:+.4f}") for k, v in w_info["meta_coefs"].items()],
            style={"display": "flex", "gap": "12px", "flexWrap": "wrap", "marginBottom": "12px"},
        ),
        graph(fig_w),
        html.H3(
            "Meta-learner regularisation",
            style={"marginTop": "20px", "marginBottom": "8px", "color": "#334155"},
        ),
        html.Div(
            [
                metric("Best meta setting", str(meta_summary["best_meta"])),
                metric("CV accuracy", f"{meta_summary['best_cv_accuracy']:.3%}"),
            ],
            style={"display": "flex", "gap": "12px", "flexWrap": "wrap", "marginBottom": "12px"},
        ),
        graph(fig_meta),
        text_box(
            steps=[
                "Correlation of OOF mistake indicators measures whether bases fail on the same examples - lower off-diagonal correlation implies more complementary errors (§16.1).",
            ],
            title="Diversity of base errors",
        ),
        html.Div(
            [metric("Mean |off-diagonal| corr.", f"{div_info['mean_abs_offdiag']:.3f}")],
            style={"display": "flex", "gap": "12px", "flexWrap": "wrap", "marginBottom": "12px"},
        ),
        graph(fig_div),
        text_box(
            steps=[
                "VotingRegressor averages base predictions; StackingRegressor learns a ridge meta-model on OOF outputs (§16.2).",
            ],
            title="Regression: voting vs stacking",
        ),
        html.Div(
            [
                metric("Best method", str(reg_stack["best_method"])),
                metric("CV R²", f"{reg_stack['best_r2']:.4f}"),
            ],
            style={"display": "flex", "gap": "12px", "flexWrap": "wrap", "marginBottom": "12px"},
        ),
        graph(fig_reg),
    ])
