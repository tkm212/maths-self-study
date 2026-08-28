"""Body content for the bar types page."""

from __future__ import annotations

from dash import html

from maths_self_study.dashboards.components import graph, graph_row, table, text_box
from maths_self_study.demos.financial_machine_learning import ch2 as helpers
from maths_self_study.demos.financial_machine_learning.data import load_ticks, project_paths


def render_body(tick_threshold, target_bars, save_flag) -> html.Div:
    try:
        ticks = load_ticks()
    except FileNotFoundError as exc:
        return text_box(steps=[str(exc)], title="Data required")

    suite = helpers.generate_bars(
        ticks,
        tick_threshold=int(tick_threshold or 100),
        target_bars=int(target_bars or 300),
    )
    if int(save_flag or 0) >= 1:
        _, _, outputs = project_paths()
        helpers.save_bars(suite, outputs)

    price_fig = helpers.plot_bar_prices(suite)
    ret_fig = helpers.plot_return_histograms(suite)
    rows = helpers.summarize_bars(suite)

    return html.Div([
        text_box(
            steps=[
                "Load tick data (Price, Quantity, time) from inputs/btc_bid_ask_data.csv.",
                "Time bars: aggregate to fixed intervals (1 second).",
                "Tick bars: close after a fixed number of transactions.",
                "Volume / dollar bars: close when cumulative quantity or dollar value hits a threshold.",
                "Compare return distributions — information-driven bars should look closer to Gaussian.",
            ],
            title="Information-driven bars (Ch. 2)",
        ),
        graph_row(graph(price_fig, style={"flex": "1"}), graph(ret_fig, style={"flex": "1"})),
        table(["Bar type", "Bars", "Returns", "Return std"], rows, caption="Bar counts and return volatility"),
    ])
