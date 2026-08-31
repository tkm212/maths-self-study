"""Shared setup and modeling helpers for ESL Chapter 2 (supervised learning) notebooks."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from maths_self_study.data import notebooks as _notebooks
from maths_self_study.viz.graphs import apply_layout, line_chart, scatter_chart, train_test_chart

ensure_package_on_path = _notebooks.ensure_package_on_path
find_project_root = _notebooks.find_project_root
init_paths = _notebooks.init_paths
load_tmdb_xy = _notebooks.load_tmdb_xy


def fit_linear_train_test_mse(
    X: pd.DataFrame,
    y: pd.Series,
    *,
    test_size: float = 0.25,
    random_state: int = 0,
) -> dict[str, Any]:
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    lin = LinearRegression()
    lin.fit(X_train, y_train)
    train_mse = mean_squared_error(y_train, lin.predict(X_train))
    test_mse = mean_squared_error(y_test, lin.predict(X_test))
    return {
        "model": lin,
        "train_mse": train_mse,
        "test_mse": test_mse,
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
    }


def subsample_xy(
    X: pd.DataFrame,
    y: pd.Series,
    max_rows: int,
    *,
    random_state: int = 0,
) -> tuple[pd.DataFrame, pd.Series]:
    if len(X) <= max_rows:
        return X, y
    idx = np.random.default_rng(random_state).choice(len(X), size=max_rows, replace=False)
    return X.iloc[idx].reset_index(drop=True), y.iloc[idx].reset_index(drop=True)


def knn_train_test_mse_figure(
    X: pd.DataFrame,
    y: pd.Series,
    *,
    ks: np.ndarray | None = None,
    max_rows: int = 40_000,
    test_size: float = 0.25,
    random_state: int = 0,
) -> tuple[go.Figure, dict[str, Any]]:
    """Scale once per split, scan $k$, plot train vs test MSE; return figure and summary dict."""
    X_run, y_run = subsample_xy(X, y, max_rows, random_state=random_state)
    X_train, X_test, y_train, y_test = train_test_split(X_run, y_run, test_size=test_size, random_state=random_state)

    lin = LinearRegression().fit(X_train, y_train)
    y_te_lin = lin.predict(X_test)

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    if ks is None:
        ks = np.arange(2, 10)
    train_mse: list[float] = []
    test_mse: list[float] = []
    for k in ks:
        knn = KNeighborsRegressor(n_neighbors=int(k), n_jobs=-1)
        knn.fit(X_train_s, y_train)
        train_mse.append(mean_squared_error(y_train, knn.predict(X_train_s)))
        test_mse.append(mean_squared_error(y_test, knn.predict(X_test_s)))

    fig = train_test_chart(
        ks,
        train_mse,
        test_mse,
        train_name="train MSE",
        test_name="test MSE",
        mode="lines+markers",
        title="$k$-NN: train vs test MSE",
        xaxis_title="k (neighbors)",
        yaxis_title="MSE",
        legend="horizontal",
    )

    k_best = int(ks[int(np.argmin(test_mse))])
    summary = {
        "k_best": k_best,
        "min_test_mse": min(test_mse),
        "linear_test_mse": float(mean_squared_error(y_test, y_te_lin)),
        "ks": ks,
        "X_train": X_train,
        "y_train": y_train,
    }
    return fig, summary


def plot_predicted_vs_actual(
    y_true: pd.Series,
    y_pred: np.ndarray,
    *,
    title: str = "Predicted vs actual",
    label: str = "test",
) -> go.Figure:
    """Scatter of predicted vs actual with a perfect-fit diagonal."""
    lo = float(min(y_true.min(), y_pred.min()))
    hi = float(max(y_true.max(), y_pred.max()))

    fig = scatter_chart(y_true, y_pred, name=label, marker_size=5, marker_opacity=0.5)
    line_chart([lo, hi], [lo, hi], mode="lines", name="perfect fit", line_dash="dash", color="black", fig=fig)
    apply_layout(
        fig,
        title=title,
        xaxis_title="actual",
        yaxis_title="predicted",
        legend="horizontal",
    )
    return fig


def default_single_feature(X: pd.DataFrame) -> str:
    if "budget" in X.columns:
        return "budget"
    if "popularity" in X.columns:
        return "popularity"
    return str(X.columns[0])


def linear_vs_knn_single_feature_figure(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    target_name: str,
    *,
    feat: str | None = None,
    k_neighbors: int = 15,
    grid_points: int = 200,
) -> tuple[go.Figure, str]:
    col = feat if feat is not None else default_single_feature(X_train)
    x1 = X_train[col].values.ravel()
    order = np.argsort(x1)
    grid = np.linspace(float(np.min(x1)), float(np.max(x1)), grid_points).reshape(-1, 1)

    lin1 = LinearRegression().fit(X_train[[col]], y_train)
    knn1 = make_pipeline(
        StandardScaler(),
        KNeighborsRegressor(n_neighbors=k_neighbors, n_jobs=-1),
    ).fit(X_train[[col]], y_train)

    fig = scatter_chart(x1[order], y_train.iloc[order], name="train", marker_size=4, marker_opacity=0.35)
    line_chart(grid.ravel(), lin1.predict(grid), mode="lines", name="linear", fig=fig)
    line_chart(grid.ravel(), knn1.predict(grid), mode="lines", name=f"k-NN (k={k_neighbors})", fig=fig)
    apply_layout(fig, title=f"Response vs {col}", xaxis_title=col, yaxis_title=target_name)
    return fig, col
