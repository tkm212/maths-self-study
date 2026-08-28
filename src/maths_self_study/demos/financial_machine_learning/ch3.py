"""Shared helpers for AFML Ch. 3 (Labeling) dashboards."""

from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go

from maths_self_study.quant.filters import cusum_filter
from maths_self_study.quant.labeling import triple_barrier_labels
from maths_self_study.viz.graphs import apply_layout

LABEL_NAMES = {-1: "Stop loss", 0: "Time out", 1: "Profit take"}
LABEL_COLORS = {1: "green", -1: "red", 0: "gray"}


def compute_labels(
    bars: pd.DataFrame,
    *,
    cusum_threshold: float = 0.0002,
    pt: float = 0.001,
    sl: float = 0.001,
    num_bars: int = 30,
) -> pd.DataFrame:
    close = bars.set_index("datetime")["close"].dropna()
    events = cusum_filter(close, threshold=float(cusum_threshold))
    return triple_barrier_labels(
        bars,
        events,
        pt=float(pt),
        sl=float(sl),
        num_bars=int(num_bars),
    )


def summarize_labels(labels: pd.DataFrame) -> list[list[str | float | int]]:
    counts = labels["label"].value_counts().sort_index()
    rows = [["Total events", f"{len(labels):,}"]]
    for label, count in counts.items():
        rows.append([LABEL_NAMES.get(int(label), str(label)), f"{count:,}"])
    return rows


def plot_triple_barrier_sample(
    bars: pd.DataFrame,
    labels: pd.DataFrame,
    *,
    n_events: int = 50,
) -> go.Figure:
    sample = labels.head(max(1, int(n_events)))
    start = sample["datetime"].min()
    end = sample["exit_time"].max()
    window_bars = bars[(bars["datetime"] >= start) & (bars["datetime"] <= end)]
    bars_idx = bars.set_index("datetime")

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=window_bars["datetime"], y=window_bars["close"], name="Close", mode="lines"))
    for _, row in sample.iterrows():
        try:
            entry_price = bars_idx.loc[row["datetime"], "close"]
            if isinstance(entry_price, pd.Series):
                entry_price = entry_price.iloc[0]
        except KeyError:
            continue
        fig.add_trace(
            go.Scatter(
                x=[row["datetime"], row["exit_time"]],
                y=[entry_price, row["exit_price"]],
                mode="lines+markers",
                line={"dash": "dot", "width": 1},
                marker={"size": 6, "color": LABEL_COLORS[row["label"]]},
                name=f"Label {int(row['label'])}",
                showlegend=False,
            )
        )
    apply_layout(
        fig,
        title="Triple-Barrier Labels on BTC 1s Bars (sample)",
        xaxis_title="Time",
        yaxis_title="Price",
        height=450,
    )
    return fig


def plot_label_distribution(labels: pd.DataFrame) -> go.Figure:
    counts = labels["label"].value_counts().sort_index()
    fig = go.Figure(
        data=[
            go.Bar(
                x=counts.index.astype(str),
                y=counts.values,
                text=counts.values,
                textposition="auto",
            )
        ]
    )
    apply_layout(
        fig,
        title="Triple-Barrier Label Distribution",
        xaxis_title="Label",
        yaxis_title="Count",
        xaxis={"tickvals": ["-1", "0", "1"], "ticktext": ["Stop loss", "Time out", "Profit take"]},
        height=400,
    )
    return fig
