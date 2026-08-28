"""Shared helpers for AFML Ch. 2 (Financial Data Structures) dashboards."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from numpy.linalg import eigh
from plotly.subplots import make_subplots

from maths_self_study.quant.bars import dollar_bars, tick_bars, time_bars, volume_bars
from maths_self_study.quant.filters import cusum_filter
from maths_self_study.viz.graphs import add_vline, apply_layout, histogram_chart


@dataclass(frozen=True)
class BarSuite:
    time_bars: pd.DataFrame
    tick_bars: pd.DataFrame
    volume_bars: pd.DataFrame
    dollar_bars: pd.DataFrame


def generate_bars(
    ticks: pd.DataFrame,
    *,
    tick_threshold: int = 100,
    target_bars: int = 300,
) -> BarSuite:
    """Build time, tick, volume, and dollar bars from tick data."""
    n_target = max(50, int(target_bars))
    tick_threshold = max(10, int(tick_threshold))

    time_df = time_bars(ticks)
    time_df = time_df[time_df["datetime"].dt.year > 2000]

    tick_df = tick_bars(ticks, threshold=tick_threshold)
    vol_threshold = ticks["Quantity"].sum() / n_target
    volume_df = volume_bars(ticks, threshold=vol_threshold)
    dollar_threshold = (ticks["Price"] * ticks["Quantity"]).sum() / n_target
    dollar_df = dollar_bars(ticks, threshold=dollar_threshold)

    return BarSuite(
        time_bars=time_df,
        tick_bars=tick_df,
        volume_bars=volume_df,
        dollar_bars=dollar_df,
    )


def save_bars(suite: BarSuite, outputs_dir) -> None:
    """Persist generated bar types to ``outputs/``."""
    outputs_dir.mkdir(parents=True, exist_ok=True)
    suite.time_bars.to_csv(outputs_dir / "btc_bid_ask_data_1s.csv", index=False)
    suite.tick_bars.to_csv(outputs_dir / "btc_bid_ask_data_tick_bars.csv", index=False)
    suite.volume_bars.to_csv(outputs_dir / "btc_bid_ask_data_volume_bars.csv", index=False)
    suite.dollar_bars.to_csv(outputs_dir / "btc_bid_ask_data_dollar_bars.csv", index=False)


def _bar_returns(bars: pd.DataFrame) -> pd.Series:
    return bars["close"].pct_change().dropna()


def summarize_bars(suite: BarSuite) -> list[list[str | float | int]]:
    rows: list[list[str | float | int]] = []
    for name, bars in [
        ("Time bars (1s)", suite.time_bars),
        ("Tick bars", suite.tick_bars),
        ("Volume bars", suite.volume_bars),
        ("Dollar bars", suite.dollar_bars),
    ]:
        ret = _bar_returns(bars)
        rows.append([name, f"{len(bars):,}", f"{len(ret):,}", f"{ret.std():.6f}"])
    return rows


def plot_bar_prices(suite: BarSuite) -> go.Figure:
    fig = make_subplots(
        rows=2,
        cols=2,
        subplot_titles=("Time bars (1s)", "Tick bars", "Volume bars", "Dollar bars (info)"),
        shared_yaxes=True,
        vertical_spacing=0.12,
        horizontal_spacing=0.08,
    )
    for i, (name, bars) in enumerate([
        ("Time bars (1s)", suite.time_bars),
        ("Tick bars", suite.tick_bars),
        ("Volume bars", suite.volume_bars),
        ("Dollar bars (info)", suite.dollar_bars),
    ]):
        t = pd.to_datetime(bars["datetime"]) if bars["datetime"].dtype != "datetime64[ns]" else bars["datetime"]
        r, c = (i // 2 + 1, i % 2 + 1)
        fig.add_trace(go.Scatter(x=t, y=bars["close"], name=name, mode="lines", line={"width": 1}), row=r, col=c)
    apply_layout(fig, height=500, title_text="BTC/USDT Bar Types (López de Prado)", showlegend=False)
    fig.update_xaxes(tickangle=-45)
    return fig


def plot_return_histograms(suite: BarSuite) -> go.Figure:
    fig = make_subplots(
        rows=2,
        cols=2,
        subplot_titles=[
            f"{n} (n={len(_bar_returns(b)):,})"
            for n, b in [
                ("Time bars", suite.time_bars),
                ("Tick bars", suite.tick_bars),
                ("Volume bars", suite.volume_bars),
                ("Dollar bars", suite.dollar_bars),
            ]
        ],
        vertical_spacing=0.12,
        horizontal_spacing=0.08,
    )
    for i, (name, bars) in enumerate([
        ("Time bars", suite.time_bars),
        ("Tick bars", suite.tick_bars),
        ("Volume bars", suite.volume_bars),
        ("Dollar bars", suite.dollar_bars),
    ]):
        ret = _bar_returns(bars)
        r, c = (i // 2 + 1, i % 2 + 1)
        histogram_chart(
            ret,
            nbinsx=50,
            histnorm="probability density",
            name=name,
            showlegend=False,
            fig=fig,
            row=r,
            col=c,
        )
        add_vline(fig, 0, line_dash="dash", line_color="red", row=r, col=c)
    apply_layout(fig, height=500, title_text="Log returns distribution by bar type")
    fig.update_xaxes(title_text="Return")
    fig.update_yaxes(title_text="Density")
    return fig


def run_cusum(close: pd.Series, *, threshold: float) -> pd.DatetimeIndex:
    return cusum_filter(close, threshold=float(threshold))


def summarize_cusum(close: pd.Series, events: pd.DatetimeIndex, *, threshold: float) -> list[list[str | float | int]]:
    return [
        ["Bars", f"{len(close):,}"],
        ["CUSUM threshold", f"{threshold:.6f}"],
        ["Events triggered", f"{len(events):,}"],
        ["Event rate", f"{100 * len(events) / max(len(close), 1):.3f}%"],
    ]


def plot_cusum_events(close: pd.Series, events: pd.DatetimeIndex, *, threshold: float) -> go.Figure:
    log_close = pd.Series(np.log(close.astype(float)), index=close.index)
    log_ret = log_close.diff().dropna()
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.7, 0.3], vertical_spacing=0.05)
    fig.add_trace(
        go.Scatter(x=close.index, y=close.values, name="Close", mode="lines", line={"width": 1}),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=events,
            y=close.reindex(events).values,
            name="CUSUM events",
            mode="markers",
            marker={"size": 8, "color": "red"},
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(x=log_ret.index, y=log_ret.values, fill="tozeroy", name="Log return", line={"width": 0}),
        row=2,
        col=1,
    )
    fig.add_hline(y=threshold, line_dash="dash", line_color="red", opacity=0.7, row=2, col=1)
    fig.add_hline(y=-threshold, line_dash="dash", line_color="red", opacity=0.7, row=2, col=1)
    apply_layout(fig, height=500, title_text="BTC Close with CUSUM Events (Snippet 2.4)")
    fig.update_yaxes(title_text="Price", row=1, col=1)
    fig.update_yaxes(title_text="Log return", row=2, col=1)
    fig.update_xaxes(title_text="Time", row=2, col=1)
    return fig


def multi_horizon_returns(bars: pd.DataFrame) -> pd.DataFrame:
    """Multi-horizon returns used as PCA features (1, 5, 10, 30 bars)."""
    ordered = bars.sort_values("datetime").reset_index(drop=True)
    close = ordered["close"].astype(float)
    rets = pd.DataFrame({
        "1-bar": close.pct_change(1),
        "5-bar": close.pct_change(5),
        "10-bar": close.pct_change(10),
        "30-bar": close.pct_change(30),
    })
    rets.index = ordered["datetime"]
    return rets.dropna()


def pca_on_returns(rets: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, list[str]]:
    corr = rets.corr()
    eigenvalues, eigenvectors = eigh(corr)
    idx = eigenvalues.argsort()[::-1]
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]
    return eigenvalues, eigenvectors, list(corr.columns)


def pc_loadings(eigenvectors: np.ndarray, columns: list[str], *, component: int = 0) -> pd.Series:
    index = max(0, min(int(component), eigenvectors.shape[1] - 1))
    weights = pd.Series(eigenvectors[:, index], index=columns)
    return weights / weights.abs().sum()


def summarize_pca(eigenvalues: np.ndarray, weights: pd.Series, *, component: int) -> list[list[str | float | int]]:
    total = float(eigenvalues.sum()) or 1.0
    idx = max(0, min(int(component), len(eigenvalues) - 1))
    rows = [[f"PC{idx + 1} variance share", f"{100 * eigenvalues[idx] / total:.1f}%"]]
    for name, weight in weights.items():
        rows.append([f"Loading: {name}", f"{weight:.4f}"])
    return rows


def plot_pca_weights(eigenvalues: np.ndarray, weights: pd.Series, *, component: int = 0) -> go.Figure:
    idx = max(0, min(int(component), len(eigenvalues) - 1))
    fig = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=(f"PCA Weights (PC {idx + 1})", "Eigenvalues (explained variance)"),
    )
    fig.add_trace(go.Bar(x=weights.index, y=weights.values, name="Weights", showlegend=False), row=1, col=1)
    fig.add_trace(
        go.Bar(x=list(range(len(eigenvalues))), y=eigenvalues, name="Eigenvalues", showlegend=False),
        row=1,
        col=2,
    )
    fig.add_hline(y=0, line_color="black", row=1, col=1)
    apply_layout(fig, height=400)
    fig.update_xaxes(title_text="Component", row=1, col=2)
    fig.update_yaxes(title_text="Weight", row=1, col=1)
    fig.update_yaxes(title_text="Eigenvalue", row=1, col=2)
    return fig
