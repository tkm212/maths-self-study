"""Shared helpers for AFML Ch. 4 (Sample Weights) dashboards."""

from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from maths_self_study.demos.financial_machine_learning import ch3 as ch3_helpers
from maths_self_study.quant.weights import (
    average_uniqueness,
    concurrent_labels_per_bar,
    time_decay_weights,
)
from maths_self_study.viz.graphs import apply_layout, histogram_chart


def compute_weighted_labels(
    bars: pd.DataFrame,
    *,
    cusum_threshold: float = 0.0002,
    pt: float = 0.001,
    sl: float = 0.001,
    num_bars: int = 30,
    decay_hours: float = 1.0,
) -> tuple[pd.DataFrame, pd.Series]:
    labels = ch3_helpers.compute_labels(
        bars,
        cusum_threshold=cusum_threshold,
        pt=pt,
        sl=sl,
        num_bars=num_bars,
    )
    t0 = labels["datetime"].min()
    t1 = labels["exit_time"].max()
    bar_index = pd.DatetimeIndex(
        bars.loc[(bars["datetime"] >= t0) & (bars["datetime"] <= t1), "datetime"].unique()
    ).sort_values()

    conc = concurrent_labels_per_bar(labels, bar_index)
    uniq = average_uniqueness(labels, bar_index)
    labels_1 = labels.assign(avg_uniqueness=uniq.values)

    ref = bars["datetime"].max()
    decay_span = pd.Timedelta(hours=float(decay_hours))
    td = time_decay_weights(labels_1["datetime"], ref_time=ref, decay_span=decay_span)
    labels_2 = labels_1.assign(time_decay=td.values)
    raw = labels_2["avg_uniqueness"] * labels_2["time_decay"]
    labels_2["sample_weight"] = raw / raw.mean()
    return labels_2, conc


def summarize_weights(labels: pd.DataFrame, *, max_concurrency: int | None = None) -> list[list[str | float | int]]:
    rows = [
        ["Mean uniqueness", f"{labels['avg_uniqueness'].mean():.4f}"],
        ["Min uniqueness", f"{labels['avg_uniqueness'].min():.4f}"],
        ["Mean time decay", f"{labels['time_decay'].mean():.4f}"],
        ["Mean sample weight", f"{labels['sample_weight'].mean():.4f}"],
    ]
    if max_concurrency is not None:
        rows.append(["Max concurrency", f"{max_concurrency:,}"])
    return rows


def plot_concurrency(
    bars: pd.DataFrame,
    conc: pd.Series,
    *,
    max_points: int = 5000,
) -> go.Figure:
    plot_conc = conc.iloc[: min(max_points, len(conc))]
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.55, 0.45], vertical_spacing=0.08)
    price_win = bars[(bars["datetime"] >= plot_conc.index.min()) & (bars["datetime"] <= plot_conc.index.max())]
    fig.add_trace(
        go.Scatter(x=price_win["datetime"], y=price_win["close"], name="Close", line={"width": 1}),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=plot_conc.index,
            y=plot_conc.values,
            name="Concurrency",
            fill="tozeroy",
            line={"width": 0},
        ),
        row=2,
        col=1,
    )
    apply_layout(fig, height=520, title_text="BTC close and label concurrency")
    fig.update_yaxes(title_text="Price", row=1, col=1)
    fig.update_yaxes(title_text="Active labels", row=2, col=1)
    return fig


def plot_uniqueness_histogram(labels: pd.DataFrame) -> go.Figure:
    return histogram_chart(
        labels["avg_uniqueness"],
        nbinsx=50,
        name="Avg uniqueness",
        title="Distribution of average uniqueness per event",
        xaxis_title="Uniqueness",
        yaxis_title="Count",
        height=400,
    )


def plot_sample_weight_histogram(labels: pd.DataFrame) -> go.Figure:
    return histogram_chart(
        labels["sample_weight"],
        nbinsx=50,
        name="Combined weight",
        title="Combined sample weights (uniqueness x time decay, mean-normalized)",
        xaxis_title="Weight",
        yaxis_title="Count",
        height=400,
    )
