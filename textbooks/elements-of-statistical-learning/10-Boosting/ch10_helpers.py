"""Shared helpers for ESL Chapter 10 (Boosting and Additive Trees) notebooks."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from sklearn.ensemble import AdaBoostClassifier, GradientBoostingRegressor
from sklearn.metrics import accuracy_score, mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

from maths_self_study.data import notebooks as _notebooks
from maths_self_study.viz.graphs import add_vline, apply_layout, histogram_chart, line_chart

init_paths = _notebooks.init_paths
load_tmdb_classification_xy = _notebooks.load_tmdb_classification_xy
load_tmdb_xy = _notebooks.load_tmdb_xy


def _select_features(
    X: pd.DataFrame,
    y: pd.Series,
    feats: list[str] | None = None,
    *,
    max_rows: int = 3000,
    random_state: int = 0,
    log_transform: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    if feats is None:
        feats = ["budget", "popularity", "runtime", "vote_average", "vote_count"]
    feats = [f for f in feats if f in X.columns]

    if len(X) > max_rows:
        rng = np.random.default_rng(random_state)
        idx = rng.choice(len(X), size=max_rows, replace=False)
        X, y = X.iloc[idx], y.iloc[idx]

    x_arr = X[feats].values.astype(float)
    y_arr = np.asarray(y, dtype=float)
    if log_transform:
        x_arr = np.log1p(np.maximum(x_arr, 0))
        y_arr = np.log1p(np.maximum(y_arr, 0))
    return x_arr, y_arr


# ---------------------------------------------------------------------------
# AdaBoost training curve (§10.1)
# ---------------------------------------------------------------------------


def adaboost_training_curve_figure(
    X: pd.DataFrame,
    y: pd.Series,
    feats: list[str] | None = None,
    *,
    n_estimators: int = 200,
    max_rows: int = 2000,
) -> tuple[go.Figure, dict[str, Any]]:
    """
    AdaBoost M1 train/test error rate vs boosting rounds (§10.1).

    AdaBoost sequentially fits weak classifiers (stumps), upweighting misclassified
    examples at each round.  The final classifier is a weighted majority vote:
        C(x) = sign[sum_m a_m G_m(x)]

    Both training and test errors are plotted at each round using the staged_predict API.
    """
    if feats is None:
        feats = ["budget", "popularity", "runtime", "vote_average", "vote_count"]
    feats = [f for f in feats if f in X.columns]

    _, _ = _select_features(X, y, feats, max_rows=max_rows, log_transform=False)
    y_cls_int = np.asarray(y, dtype=int).ravel()
    if len(X) > max_rows:
        rng = np.random.default_rng(0)
        idx = rng.choice(len(X), size=max_rows, replace=False)
        x_arr = np.log1p(np.maximum(X.iloc[idx][feats].values.astype(float), 0))
        y_cls_int = y_cls_int[idx]
    else:
        x_arr = np.log1p(np.maximum(X[feats].values.astype(float), 0))

    X_tr, X_te, y_tr, y_te = train_test_split(x_arr, y_cls_int, test_size=0.25, random_state=0)

    clf = AdaBoostClassifier(
        estimator=DecisionTreeClassifier(max_depth=1),
        n_estimators=n_estimators,
        random_state=0,
    )
    clf.fit(X_tr, y_tr)

    train_errors, test_errors = [], []
    for y_pred_tr, y_pred_te in zip(clf.staged_predict(X_tr), clf.staged_predict(X_te), strict=False):
        train_errors.append(float(1 - accuracy_score(y_tr, y_pred_tr)))
        test_errors.append(float(1 - accuracy_score(y_te, y_pred_te)))

    rounds = list(range(1, len(train_errors) + 1))
    best_round = rounds[int(np.argmin(test_errors))]

    fig = line_chart(rounds, train_errors, mode="lines", name="Train error rate", color="steelblue")
    line_chart(rounds, test_errors, mode="lines", name="Test error rate", color="tomato", fig=fig)
    add_vline(fig, x=best_round, line_dash="dash", line_color="grey", annotation_text=f"best round={best_round}")
    apply_layout(
        fig,
        title=f"AdaBoost training curve (stumps, {n_estimators} rounds) — §10.1",
        xaxis_title="boosting round M",
        yaxis_title="misclassification rate",
        legend="horizontal",
    )
    return fig, {
        "best_round": best_round,
        "best_test_error": min(test_errors),
        "final_train_error": train_errors[-1],
    }


def margin_distribution_figure(
    X: pd.DataFrame,
    y: pd.Series,
    feats: list[str] | None = None,
    *,
    n_estimators_list: list[int] | None = None,
    max_rows: int = 2000,
) -> go.Figure:
    """
    Distribution of AdaBoost margins at different numbers of rounds (§10.4, §10.6).

    The **margin** of an observation is y_i * F(x_i) / sum_m |a_m|.  A positive margin
    means correct classification; the magnitude measures confidence.  As boosting
    continues, the margin distribution shifts right -- AdaBoost maximises a soft margin.
    This connection to the exponential loss (§10.4) is key to understanding boosting.
    """
    if n_estimators_list is None:
        n_estimators_list = [10, 50, 200]
    if feats is None:
        feats = ["budget", "popularity", "runtime", "vote_average", "vote_count"]
    feats = [f for f in feats if f in X.columns]

    if len(X) > max_rows:
        rng = np.random.default_rng(0)
        idx = rng.choice(len(X), size=max_rows, replace=False)
        x_arr = np.log1p(np.maximum(X.iloc[idx][feats].values.astype(float), 0))
        y_cls = np.asarray(y, dtype=int).ravel()[idx]
    else:
        x_arr = np.log1p(np.maximum(X[feats].values.astype(float), 0))
        y_cls = np.asarray(y, dtype=int).ravel()

    X_tr, X_te, y_tr, y_te = train_test_split(x_arr, y_cls, test_size=0.25, random_state=0)
    y_te_signed = np.where(y_te == 1, 1, -1).astype(float)

    colors = ["steelblue", "tomato", "green", "purple"]

    def _margin_histogram(fig: go.Figure | None, m: int, color: str) -> go.Figure:
        clf = AdaBoostClassifier(
            estimator=DecisionTreeClassifier(max_depth=1),
            n_estimators=m,
            random_state=0,
        )
        clf.fit(X_tr, y_tr)
        decision = clf.decision_function(X_te)
        norm = float(np.sum(np.abs(clf.estimator_weights_)))
        margins = y_te_signed * decision / (norm + 1e-10)
        return histogram_chart(
            margins,
            name=f"M={m}",
            opacity=0.5,
            nbinsx=40,
            color=color,
            fig=fig,
        )

    fig = _margin_histogram(None, n_estimators_list[0], colors[0])
    for i, m in enumerate(n_estimators_list[1:], start=1):
        fig = _margin_histogram(fig, m, colors[i % len(colors)])
    add_vline(fig, x=0, line_dash="dash", line_color="black", annotation_text="margin=0")
    apply_layout(
        fig,
        title="AdaBoost margin distribution at increasing rounds — §10.4",
        xaxis_title="margin  y·F(x) / Σ|αₘ|",
        yaxis_title="count",
        barmode="overlay",
        legend="horizontal",
    )
    return fig


# ---------------------------------------------------------------------------
# Gradient boosting (§10.9-10.12)
# ---------------------------------------------------------------------------


def gbm_n_estimators_figure(
    X: pd.DataFrame,
    y: pd.Series,
    feats: list[str] | None = None,
    *,
    n_estimators: int = 200,
    learning_rate: float = 0.1,
    max_depth: int = 3,
    max_rows: int = 2000,
) -> tuple[go.Figure, dict[str, Any]]:
    """
    Gradient boosting train/test MSE vs number of trees (§10.9).

    GBM fits each successive tree to the **negative gradient** of the loss:
        F_m(x) = F_{m-1}(x) + nu * gamma_m * h(x; a_m)
    where h(x; a_m) minimises the sum of squared pseudo-residuals
        r_{im} = -[dL(y_i, F(x_i)) / dF(x_i)]_{F = F_{m-1}}

    For squared error loss the pseudo-residuals are just ordinary residuals.
    Uses sklearn's staged_predict to track error at every stage.
    """
    if feats is None:
        feats = ["budget", "popularity", "runtime", "vote_average", "vote_count"]
    feats = [f for f in feats if f in X.columns]

    x_arr, y_log = _select_features(X, y, feats, max_rows=max_rows, log_transform=True)
    X_tr, X_te, y_tr, y_te = train_test_split(x_arr, y_log, test_size=0.25, random_state=0)

    gbm = GradientBoostingRegressor(
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        max_depth=max_depth,
        random_state=0,
    )
    gbm.fit(X_tr, y_tr)

    train_mse, test_mse = [], []
    for y_tr_pred, y_te_pred in zip(gbm.staged_predict(X_tr), gbm.staged_predict(X_te), strict=False):
        train_mse.append(float(mean_squared_error(y_tr, y_tr_pred)))
        test_mse.append(float(mean_squared_error(y_te, y_te_pred)))

    rounds = list(range(1, len(train_mse) + 1))
    best_round = rounds[int(np.argmin(test_mse))]

    fig = line_chart(rounds, train_mse, mode="lines", name="Train MSE", color="steelblue")
    line_chart(rounds, test_mse, mode="lines", name="Test MSE", color="tomato", fig=fig)
    add_vline(fig, x=best_round, line_dash="dash", line_color="grey", annotation_text=f"best M={best_round}")
    apply_layout(
        fig,
        title=f"GBM MSE vs rounds (nu={learning_rate}, depth={max_depth}) — §10.9",
        xaxis_title="number of trees M",
        yaxis_title="MSE (log₁p-space)",
        legend="horizontal",
    )
    return fig, {"best_round": best_round, "best_test_mse": min(test_mse)}


def gbm_shrinkage_figure(
    X: pd.DataFrame,
    y: pd.Series,
    feats: list[str] | None = None,
    *,
    n_estimators: int = 300,
    learning_rates: list[float] | None = None,
    max_depth: int = 3,
    max_rows: int = 2000,
) -> go.Figure:
    """
    Gradient boosting shrinkage: test MSE vs rounds for different learning rates (§10.12).

    The **shrinkage** parameter nu in (0, 1] scales each tree's contribution:
        F_m(x) = F_{m-1}(x) + nu * gamma_m * h(x; a_m)

    Smaller nu requires more trees (M must increase inversely) but typically yields a better
    regularised solution -- trading computation for accuracy.  The optimal M is selected by
    the minimum test-error curve for each nu.
    """
    if learning_rates is None:
        learning_rates = [0.5, 0.1, 0.01]
    if feats is None:
        feats = ["budget", "popularity", "runtime", "vote_average", "vote_count"]
    feats = [f for f in feats if f in X.columns]

    x_arr, y_log = _select_features(X, y, feats, max_rows=max_rows, log_transform=True)
    X_tr, X_te, y_tr, y_te = train_test_split(x_arr, y_log, test_size=0.25, random_state=0)

    colors = ["steelblue", "tomato", "green", "purple"]

    def _shrinkage_curve(fig: go.Figure | None, lr: float, color: str) -> go.Figure:
        gbm = GradientBoostingRegressor(
            n_estimators=n_estimators,
            learning_rate=lr,
            max_depth=max_depth,
            random_state=0,
        )
        gbm.fit(X_tr, y_tr)
        test_mse = [float(mean_squared_error(y_te, pred)) for pred in gbm.staged_predict(X_te)]
        rounds = list(range(1, len(test_mse) + 1))
        best_m = rounds[int(np.argmin(test_mse))]
        return line_chart(
            rounds,
            test_mse,
            mode="lines",
            name=f"nu={lr} (best M={best_m})",
            color=color,
            fig=fig,
        )

    fig = _shrinkage_curve(None, learning_rates[0], colors[0])
    for i, lr in enumerate(learning_rates[1:], start=1):
        fig = _shrinkage_curve(fig, lr, colors[i % len(colors)])
    apply_layout(
        fig,
        title=f"GBM shrinkage: test MSE vs rounds for different nu (depth={max_depth}) — §10.12",
        xaxis_title="number of trees M",
        yaxis_title="test MSE (log₁p-space)",
        legend="horizontal",
    )
    return fig


def gbm_feature_importance_figure(
    X: pd.DataFrame,
    y: pd.Series,
    feats: list[str] | None = None,
    *,
    n_estimators: int = 200,
    learning_rate: float = 0.1,
    max_depth: int = 3,
    max_rows: int = 2000,
) -> tuple[go.Figure, dict[str, Any]]:
    """
    GBM variable importance (§10.13).

    Importance of feature j is defined as the average improvement in the split criterion
    (squared error) per split on feature j, averaged over all trees:

        Ĵ_j² = (1/M) Σ_m Σ_{splits on j in tree m} improvement_t

    sklearn's ``feature_importances_`` attribute implements this directly.
    """
    if feats is None:
        feats = ["budget", "popularity", "runtime", "vote_average", "vote_count"]
    feats = [f for f in feats if f in X.columns]

    x_arr, y_log = _select_features(X, y, feats, max_rows=max_rows, log_transform=True)
    gbm = GradientBoostingRegressor(
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        max_depth=max_depth,
        random_state=0,
    )
    gbm.fit(x_arr, y_log)

    importances = gbm.feature_importances_
    order = np.argsort(importances)[::-1]
    sorted_feats = [feats[i] for i in order]
    sorted_imp = importances[order].tolist()

    fig = go.Figure(
        go.Bar(
            x=sorted_feats,
            y=sorted_imp,
            marker_color="steelblue",
        )
    )
    apply_layout(
        fig,
        title=f"GBM variable importance (M={n_estimators}, nu={learning_rate}, depth={max_depth}) — §10.13",
        xaxis_title="feature",
        yaxis_title="relative importance",
    )
    top_feat = sorted_feats[0]
    return fig, {"top_feature": top_feat, "top_importance": float(sorted_imp[0])}
