"""Shared helpers for ESL Chapter 3 (Linear Methods for Regression) notebooks."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from sklearn.cross_decomposition import PLSRegression
from sklearn.decomposition import PCA
from sklearn.linear_model import Lasso, LinearRegression, Ridge, lasso_path
from sklearn.metrics import mean_squared_error, r2_score

from maths_self_study.data import notebooks as _notebooks
from maths_self_study.viz.graphs import add_vline, apply_layout, bar_chart, line_chart, train_test_chart

find_project_root = _notebooks.find_project_root
init_paths = _notebooks.init_paths
load_tmdb_xy = _notebooks.load_tmdb_xy
scale_split = _notebooks.scale_split

# ---------------------------------------------------------------------------
# Subset selection
# ---------------------------------------------------------------------------


def forward_stepwise(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
) -> tuple[list[str], list[float], list[float]]:
    """Greedy forward stepwise: at each step add the feature that best reduces test MSE."""
    remaining = list(X_train.columns)
    selected: list[str] = []
    train_mse: list[float] = []
    test_mse: list[float] = []

    while remaining:
        best_feat: str | None = None
        best_mse = np.inf
        for feat in remaining:
            cols = [*selected, feat]
            lin = LinearRegression().fit(X_train[cols], y_train)
            mse = float(mean_squared_error(y_test, lin.predict(X_test[cols])))
            if mse < best_mse:
                best_mse = mse
                best_feat = feat
        if best_feat is None:
            break
        selected.append(best_feat)
        remaining.remove(best_feat)
        lin_full = LinearRegression().fit(X_train[selected], y_train)
        train_mse.append(float(mean_squared_error(y_train, lin_full.predict(X_train[selected]))))
        test_mse.append(float(mean_squared_error(y_test, lin_full.predict(X_test[selected]))))

    return selected, train_mse, test_mse


def subset_selection_figure(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
) -> tuple[go.Figure, list[str]]:
    selected, train_mse, test_mse = forward_stepwise(X_train, X_test, y_train, y_test)
    n = list(range(1, len(selected) + 1))
    fig = train_test_chart(
        n,
        train_mse,
        test_mse,
        train_name="train MSE",
        test_name="test MSE",
        mode="lines+markers",
        title="Forward stepwise selection: MSE vs number of features",
        xaxis_title="# features",
        yaxis_title="MSE",
        legend="horizontal",
    )
    return fig, selected


# ---------------------------------------------------------------------------
# Ridge
# ---------------------------------------------------------------------------


def ridge_alpha_path_figure(
    X_train_s: np.ndarray,
    X_test_s: np.ndarray,
    y_train: pd.Series,
    y_test: pd.Series,
    *,
    n_alphas: int = 80,
) -> tuple[go.Figure, dict[str, Any]]:
    """Fit Ridge for a log-spaced grid of alphas; return test-MSE figure and best-alpha summary."""
    alphas = np.logspace(-2, 6, n_alphas)
    train_mse, test_mse = [], []
    for alpha in alphas:
        ridge = Ridge(alpha=float(alpha))
        ridge.fit(X_train_s, y_train)
        train_mse.append(float(mean_squared_error(y_train, ridge.predict(X_train_s))))
        test_mse.append(float(mean_squared_error(y_test, ridge.predict(X_test_s))))

    best_idx = int(np.argmin(test_mse))
    fig = train_test_chart(
        alphas,
        train_mse,
        test_mse,
        train_name="train MSE",
        test_name="test MSE",
        title="Ridge: MSE vs regularisation strength (alpha)",
        xaxis_title="alpha (log scale)",
        yaxis_title="MSE",
        xaxis_type="log",
        legend="horizontal",
    )
    add_vline(
        fig,
        x=float(alphas[best_idx]),
        line_dash="dash",
        line_color="grey",
        annotation_text=f"best alpha={alphas[best_idx]:.1f}",
    )
    return fig, {"best_alpha": float(alphas[best_idx]), "min_test_mse": min(test_mse)}


def ridge_coef_figure(
    X_train_s: np.ndarray,
    y_train: pd.Series,
    feat_names: list[str],
    *,
    n_alphas: int = 80,
) -> go.Figure:
    """Coefficient shrinkage paths as alpha increases."""
    alphas = np.logspace(-2, 6, n_alphas)
    coefs = np.array([Ridge(alpha=float(a)).fit(X_train_s, y_train).coef_ for a in alphas])
    fig = line_chart(alphas, coefs[:, 0], mode="lines", name=feat_names[0])
    for i, name in enumerate(feat_names[1:], start=1):
        line_chart(alphas, coefs[:, i], mode="lines", name=name, fig=fig)
    apply_layout(
        fig,
        title="Ridge: coefficient shrinkage paths",
        xaxis_title="alpha (log scale)",
        xaxis_type="log",
        yaxis_title="coefficient",
    )
    return fig


# ---------------------------------------------------------------------------
# Lasso
# ---------------------------------------------------------------------------


def lasso_coef_path_figure(
    X_train_s: np.ndarray,
    y_train: pd.Series,
    feat_names: list[str],
) -> go.Figure:
    """Lasso coefficient paths vs log(alpha) — shows which features are selected."""
    alphas_path, coefs_path, _ = lasso_path(X_train_s, np.asarray(y_train))
    fig = line_chart(np.log10(alphas_path + 1e-10), coefs_path[0], mode="lines", name=feat_names[0])
    for i, name in enumerate(feat_names[1:], start=1):
        line_chart(np.log10(alphas_path + 1e-10), coefs_path[i], mode="lines", name=name, fig=fig)
    apply_layout(
        fig,
        title="Lasso: coefficient paths (features entering as alpha decreases)",
        xaxis_title="log10(alpha)",
        yaxis_title="coefficient",
    )
    return fig


def lasso_alpha_path_figure(
    X_train_s: np.ndarray,
    X_test_s: np.ndarray,
    y_train: pd.Series,
    y_test: pd.Series,
    *,
    n_alphas: int = 60,
) -> tuple[go.Figure, dict[str, Any]]:
    """Train/test MSE vs lasso alpha; return figure and best-alpha summary."""
    alphas = np.logspace(-2, 8, n_alphas)
    train_mse, test_mse, n_nonzero = [], [], []
    for alpha in alphas:
        lasso = Lasso(alpha=float(alpha), max_iter=10_000)
        lasso.fit(X_train_s, y_train)
        train_mse.append(float(mean_squared_error(y_train, lasso.predict(X_train_s))))
        test_mse.append(float(mean_squared_error(y_test, lasso.predict(X_test_s))))
        n_nonzero.append(int(np.sum(np.abs(lasso.coef_) > 1e-8)))

    best_idx = int(np.argmin(test_mse))
    fig = train_test_chart(
        alphas,
        train_mse,
        test_mse,
        train_name="train MSE",
        test_name="test MSE",
        mode="lines",
    )
    add_vline(
        fig,
        x=float(alphas[best_idx]),
        line_dash="dash",
        line_color="grey",
        annotation_text=f"best alpha={alphas[best_idx]:.1f}",
    )
    apply_layout(
        fig,
        title="Lasso: MSE vs regularisation strength (alpha)",
        xaxis_title="alpha (log scale)",
        xaxis_type="log",
        yaxis_title="MSE",
        legend="horizontal",
    )
    return fig, {
        "best_alpha": float(alphas[best_idx]),
        "min_test_mse": min(test_mse),
        "n_nonzero_at_best": n_nonzero[best_idx],
    }


def lasso_selected_coef_figure(
    X_train_s: np.ndarray,
    y_train: pd.Series,
    feat_names: list[str],
    best_alpha: float,
) -> go.Figure:
    """Bar chart of non-zero Lasso coefficients at the best alpha."""
    lasso = Lasso(alpha=best_alpha, max_iter=10_000)
    lasso.fit(X_train_s, y_train)
    coefs = pd.Series(lasso.coef_, index=feat_names).sort_values()
    coefs = coefs[coefs.abs() > 1e-8]
    fig = bar_chart(
        coefs.values,
        coefs.index.tolist(),
        orientation="h",
        title=f"Lasso selected features (alpha={best_alpha:.1f})",
        xaxis_title="coefficient",
    )
    return fig


# ---------------------------------------------------------------------------
# PCR and PLS (section 3.5)
# ---------------------------------------------------------------------------


def pcr_pls_figure(
    X_train_s: np.ndarray,
    X_test_s: np.ndarray,
    y_train: pd.Series,
    y_test: pd.Series,
) -> tuple[go.Figure, dict[str, Any]]:
    """Test MSE vs number of components for PCR and PLS; return figure and best-n summary."""
    n_feats = X_train_s.shape[1]
    n_range = list(range(1, n_feats + 1))

    pcr_mse: list[float] = []
    pls_mse: list[float] = []

    for n in n_range:
        pca = PCA(n_components=n)
        lin = LinearRegression()
        lin.fit(pca.fit_transform(X_train_s), y_train)
        pcr_mse.append(float(mean_squared_error(y_test, lin.predict(pca.transform(X_test_s)))))

        pls = PLSRegression(n_components=n)
        pls.fit(X_train_s, y_train)
        pls_mse.append(float(mean_squared_error(y_test, pls.predict(X_test_s))))

    best_pcr_n = int(n_range[int(np.argmin(pcr_mse))])
    best_pls_n = int(n_range[int(np.argmin(pls_mse))])

    fig = line_chart(
        n_range,
        pcr_mse,
        name="PCR",
        mode="lines+markers",
        title="PCR vs PLS: test MSE vs number of components",
        xaxis_title="# components",
        yaxis_title="test MSE",
    )
    line_chart(n_range, pls_mse, name="PLS", mode="lines+markers", fig=fig)
    return fig, {
        "best_pcr_n": best_pcr_n,
        "min_pcr_mse": min(pcr_mse),
        "best_pls_n": best_pls_n,
        "min_pls_mse": min(pls_mse),
    }


# ---------------------------------------------------------------------------
# Model comparison table (section 3.6)
# ---------------------------------------------------------------------------


def compare_models(
    X_train_s: np.ndarray,
    X_test_s: np.ndarray,
    y_train: pd.Series,
    y_test: pd.Series,
    *,
    alpha_ridge: float,
    alpha_lasso: float,
    n_pcr: int | None = None,
    n_pls: int | None = None,
) -> pd.DataFrame:
    """Compare OLS, Ridge, Lasso, and optionally PCR and PLS on test MSE and R²."""
    n_feats = X_train_s.shape[1]
    n_pcr = n_pcr or n_feats
    n_pls = n_pls or n_feats

    pca = PCA(n_components=n_pcr)
    X_train_pcr = pca.fit_transform(X_train_s)
    X_test_pcr = pca.transform(X_test_s)

    pls = PLSRegression(n_components=n_pls)
    pls.fit(X_train_s, y_train)

    rows = []
    for name, y_pred in [
        ("OLS", LinearRegression().fit(X_train_s, y_train).predict(X_test_s)),
        (f"Ridge (alpha={alpha_ridge:.0f})", Ridge(alpha=alpha_ridge).fit(X_train_s, y_train).predict(X_test_s)),
        (
            f"Lasso (alpha={alpha_lasso:.1f})",
            Lasso(alpha=alpha_lasso, max_iter=10_000).fit(X_train_s, y_train).predict(X_test_s),
        ),
        (f"PCR ({n_pcr} components)", LinearRegression().fit(X_train_pcr, y_train).predict(X_test_pcr)),
        (f"PLS ({n_pls} components)", pls.predict(X_test_s)),
    ]:
        rows.append({
            "model": name,
            "test_MSE": round(float(mean_squared_error(y_test, y_pred)), 4),
            "test_R2": round(float(r2_score(y_test, y_pred)), 4),
        })
    return pd.DataFrame(rows).set_index("model")
