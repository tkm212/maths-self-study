"""Body content for the triple-barrier page."""

from __future__ import annotations

from dash import html

from maths_self_study.dashboards.components import graph, table, text_box
from maths_self_study.demos.financial_machine_learning import ch3 as helpers
from maths_self_study.demos.financial_machine_learning.data import load_time_bars


def render_body(cusum_threshold, pt, sl, num_bars, sample_n) -> html.Div:
    try:
        bars = load_time_bars()
    except FileNotFoundError as exc:
        return text_box(steps=[str(exc)], title="Data required")

    labels = helpers.compute_labels(
        bars,
        cusum_threshold=float(cusum_threshold or 0.0002),
        pt=float(pt or 0.001),
        sl=float(sl or 0.001),
        num_bars=int(num_bars or 30),
    )
    sample_fig = helpers.plot_triple_barrier_sample(bars, labels, n_events=int(sample_n or 50))
    dist_fig = helpers.plot_label_distribution(labels)
    rows = helpers.summarize_labels(labels)

    return html.Div([
        text_box(
            steps=[
                "Run CUSUM on close to pick event times (Ch. 2).",
                "At each event, set upper (profit), lower (stop), and vertical (time) barriers.",
                "Label +1 / -1 / 0 by first barrier touched along the price path.",
                "Path-dependent labels match realized trade outcomes better than fixed-horizon returns.",
            ],
            title="Triple-barrier method (Snippet 3.2)",
        ),
        graph(sample_fig),
        graph(dist_fig),
        table(["Outcome", "Count"], rows, caption="Label summary"),
    ])
