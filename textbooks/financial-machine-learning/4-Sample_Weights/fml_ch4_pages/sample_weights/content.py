"""Body content for the sample weights page."""

from __future__ import annotations

from dash import html

from maths_self_study.dashboards.components import graph, graph_row, table, text_box
from maths_self_study.demos.financial_machine_learning import ch4 as helpers
from maths_self_study.demos.financial_machine_learning.data import load_time_bars


def render_body(cusum_threshold, pt, sl, num_bars, decay_hours) -> html.Div:
    try:
        bars = load_time_bars()
    except FileNotFoundError as exc:
        return text_box(steps=[str(exc)], title="Data required")

    labels, _conc = helpers.compute_weighted_labels(
        bars,
        cusum_threshold=float(cusum_threshold or 0.0002),
        pt=float(pt or 0.001),
        sl=float(sl or 0.001),
        num_bars=int(num_bars or 30),
        decay_hours=float(decay_hours or 1.0),
    )
    uniq_fig = helpers.plot_uniqueness_histogram(labels)
    weight_fig = helpers.plot_sample_weight_histogram(labels)
    rows = helpers.summarize_weights(labels)

    return html.Div([
        text_box(
            steps=[
                "Compute average uniqueness ūᵢ = mean of 1/c(t) over each event's life.",
                "Apply time decay w = exp(−age/τ) relative to end of sample.",
                "Combined sample_weight = normalize(ū × decay); pass to sklearn estimators.",
                "Sequential bootstrap (book) complements explicit weights in ensemble methods.",
            ],
            title="Sample weights and time decay",
        ),
        graph_row(graph(uniq_fig, style={"flex": "1"}), graph(weight_fig, style={"flex": "1"})),
        table(["Measure", "Value"], rows, caption="Weight summary"),
    ])
