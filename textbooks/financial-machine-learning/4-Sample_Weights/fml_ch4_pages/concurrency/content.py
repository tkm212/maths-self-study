"""Body content for the concurrency page."""

from __future__ import annotations

from dash import html

from maths_self_study.dashboards.components import graph, table, text_box
from maths_self_study.demos.financial_machine_learning import ch4 as helpers
from maths_self_study.demos.financial_machine_learning.data import load_time_bars


def render_body(cusum_threshold, pt, sl, num_bars, max_points) -> html.Div:
    try:
        bars = load_time_bars()
    except FileNotFoundError as exc:
        return text_box(steps=[str(exc)], title="Data required")

    labels, conc = helpers.compute_weighted_labels(
        bars,
        cusum_threshold=float(cusum_threshold or 0.0002),
        pt=float(pt or 0.001),
        sl=float(sl or 0.001),
        num_bars=int(num_bars or 30),
        decay_hours=1.0,
    )
    fig = helpers.plot_concurrency(bars, conc, max_points=int(max_points or 5000))
    rows = [
        ["Events labeled", f"{len(labels):,}"],
        ["Max concurrency", f"{int(conc.max()):,}"],
        ["Mean concurrency", f"{conc.mean():.2f}"],
    ]

    return html.Div([
        text_box(
            steps=[
                "Triple-barrier events overlap in time — labels are not IID.",
                "At each bar t, count c(t) = number of events still open.",
                "High concurrency means many labels share the same price path.",
                "Down-weight crowded periods via average uniqueness (next page).",
            ],
            title="Label concurrency",
        ),
        graph(fig),
        table(["Measure", "Value"], rows, caption="Concurrency summary"),
    ])
