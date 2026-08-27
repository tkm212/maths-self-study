"""Body content for the SGD page."""

from __future__ import annotations

from dash import html

from maths_self_study.dashboards.components import graph, table, text_box
from maths_self_study.dashboards.utils import coerce_float
from maths_self_study.demos.deep_learning import ch5 as helpers
from maths_self_study.viz.latex import formula_group
from maths_self_study.viz.textbooks.deep_learning.ch5.formulas import GD_UPDATE, MINIBATCH_UPDATE


def render_body(eta, batch_size) -> html.Div:
    learning_rate = coerce_float(eta, default=helpers.SGD_LEARNING_RATE)
    batch = int(coerce_float(batch_size, default=helpers.SGD_BATCH_SIZE))
    batch = max(1, batch)
    ctx = helpers.sgd_demo_context(batch)
    n_train = int(ctx["n_train"])
    fig = helpers.plot_sgd_paths(learning_rate, batch)

    rows = [
        ["Training examples m", str(n_train)],
        ["Mini-batch size |B|", str(ctx["batch_size"])],
        ["Gradient cost per step", f"O({ctx['batch_size']}) vs O({n_train}) full-batch"],
        ["Regime", str(ctx["regime"])],
    ]

    return html.Div([
        html.H3("Full-batch vs mini-batch descent on linear regression"),
        text_box(
            steps=[
                "Each step draws a random mini-batch B from the training set and updates "
                "weights using only those examples — not the full dataset.",
                f"Full-batch GD (|B| = {n_train}) computes the exact average gradient; the blue curve falls smoothly.",
                f"Mini-batch SGD (|B| = {batch}) uses a noisy gradient estimate; "
                "the red curve zig-zags but each step is cheaper when |B| ≪ m.",
                "Batch size = 1 is the noisiest (one random point per step). "
                "Large batches reduce noise but approach full-batch cost.",
                "Deep learning almost always uses mini-batches because they balance "
                "statistical efficiency, hardware parallelism, and memory.",
            ],
            title="What is a mini-batch?",
        ),
        formula_group(
            ("Full-batch gradient descent", GD_UPDATE),
            ("Mini-batch SGD", MINIBATCH_UPDATE),
            title="Key formulas (§5.9)",
        ),
        html.P(
            "Both curves use the same polynomial features and learning rate. "
            "Try shrinking the batch size to see more noise, or set |B| = m for full-batch behaviour.",
            style={"color": "#475569", "fontSize": "0.95rem"},
        ),
        table(["Setting", "Value"], rows, caption="Current mini-batch settings"),
        graph(fig),
    ])
