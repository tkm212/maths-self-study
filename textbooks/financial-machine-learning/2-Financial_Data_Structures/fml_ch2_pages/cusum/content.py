"""Body content for the CUSUM page."""

from __future__ import annotations

from dash import html

from maths_self_study.dashboards.components import graph, table, text_box
from maths_self_study.demos.financial_machine_learning import ch2 as helpers
from maths_self_study.demos.financial_machine_learning.data import load_time_bars


def render_body(threshold) -> html.Div:
    try:
        bars = load_time_bars()
    except FileNotFoundError as exc:
        return text_box(steps=[str(exc)], title="Data required")

    close = bars.set_index("datetime")["close"].dropna()
    threshold = float(threshold or 0.0002)
    events = helpers.run_cusum(close, threshold=threshold)
    fig = helpers.plot_cusum_events(close, events, threshold=threshold)
    rows = helpers.summarize_cusum(close, events, threshold=threshold)

    return html.Div([
        text_box(
            steps=[
                "Compute log returns on 1-second close prices.",
                "Accumulate signed divergences in S_t; fire an event when |S_t| ≥ threshold, then reset.",
                "Unlike Bollinger-style rules, hovering near a level does not spam events.",
                "Use event timestamps as seeds for triple-barrier labeling (Ch. 3).",
            ],
            title="CUSUM filter (Snippet 2.4)",
        ),
        graph(fig),
        table(["Measure", "Value"], rows, caption="CUSUM summary"),
    ])
