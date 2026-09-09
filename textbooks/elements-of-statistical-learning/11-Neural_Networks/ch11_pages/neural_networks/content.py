"""Body content for neural networks page."""

from __future__ import annotations

import ch11_helpers as helpers
from ch11_data import load_cls
from dash import html

from maths_self_study.dashboards.components import graph, metric, text_box


def render_body(max_epochs, hidden_units) -> html.Div:
    X, y, _ = load_cls()
    max_epochs = int(max_epochs or 80)
    hidden_units = int(hidden_units or 20)
    hidden_layer_sizes = (hidden_units,)
    try:
        fig_curve, curve_summary = helpers.nn_training_curve_figure(
            X, y, hidden_layer_sizes=hidden_layer_sizes, max_epochs=max_epochs
        )
        fig_wd = helpers.nn_weight_decay_figure(X, y, hidden_layer_sizes=hidden_layer_sizes, max_epochs=max_epochs)
        fig_arch, arch_summary = helpers.nn_architecture_figure(X, y)
    except FileNotFoundError as exc:
        return text_box(steps=[str(exc)], title="Data required")

    return html.Div([
        text_box(
            steps=[
                "Track cross-entropy training loss and test misclassification rate each epoch; the dashed line marks minimum test error (§11.4).",
            ],
            title="Training curve: loss and test error vs epoch",
        ),
        html.Div(
            [
                metric("Best epoch", str(curve_summary["best_epoch"])),
                metric("Best test error", f"{curve_summary['best_test_error']:.3%}"),
                metric("Final train loss", f"{curve_summary['final_train_loss']:.4f}"),
            ],
            style={"display": "flex", "gap": "12px", "flexWrap": "wrap", "marginBottom": "12px"},
        ),
        graph(fig_curve),
        text_box(
            steps=[
                r"$L^2$ penalty $\frac{\lambda}{2}\|\theta\|^2$ prevents arbitrarily large weights; sklearn `alpha` is $\lambda$ (§11.5.2).",
            ],
            title="Weight decay: regularisation by penalising large weights",
        ),
        graph(fig_wd),
        text_box(
            steps=[
                "Compare architectures by 3-fold CV accuracy — `(20,)` is one layer with 20 units; `(20, 10)` is two layers (§11.5.4).",
            ],
            title="Architecture: number of hidden units and layers",
        ),
        html.Div(
            [
                metric("Best architecture", str(arch_summary["best_arch"])),
                metric("CV accuracy", f"{arch_summary['best_cv_accuracy']:.3%}"),
            ],
            style={"display": "flex", "gap": "12px", "flexWrap": "wrap", "marginBottom": "12px"},
        ),
        graph(fig_arch),
    ])
