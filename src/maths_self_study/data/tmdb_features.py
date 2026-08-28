"""TMDB feature preparation helpers for ESL textbook notebooks."""

from __future__ import annotations

import numpy as np
import pandas as pd

DEFAULT_TMDB_FEATURES = ["budget", "popularity", "runtime", "vote_average", "vote_count"]

__all__ = [
    "DEFAULT_TMDB_FEATURES",
    "log_feature_xy",
    "log_features_xy",
    "prepare_tmdb_all_numeric_log1p",
    "prepare_tmdb_cls",
    "prepare_tmdb_features",
    "prepare_tmdb_numeric_matrix",
    "prepare_tmdb_reg",
    "prepare_tmdb_unsupervised",
    "resolve_features",
    "single_feature_subsample",
    "subsample_log1p_xy",
    "subsample_xy",
]


def resolve_features(X: pd.DataFrame, feats: list[str] | None = None) -> list[str]:
    if feats is None:
        feats = DEFAULT_TMDB_FEATURES
    return [f for f in feats if f in X.columns]


def subsample_xy(
    X: pd.DataFrame,
    y: pd.Series,
    *,
    max_rows: int,
    random_state: int = 0,
) -> tuple[pd.DataFrame, pd.Series]:
    if len(X) <= max_rows:
        return X, y
    rng = np.random.default_rng(random_state)
    idx = rng.choice(len(X), size=max_rows, replace=False)
    return X.iloc[idx], y.iloc[idx]


def prepare_tmdb_cls(
    X: pd.DataFrame,
    y: pd.Series,
    feats: list[str] | None = None,
    *,
    max_rows: int = 2000,
    random_state: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    feats = resolve_features(X, feats)
    if len(X) > max_rows:
        rng = np.random.default_rng(random_state)
        idx = rng.choice(len(X), size=max_rows, replace=False)
        x_sub = X.iloc[idx][feats].values.astype(float)
        y_sub = np.asarray(y, dtype=int).ravel()[idx]
    else:
        x_sub = X[feats].values.astype(float)
        y_sub = np.asarray(y, dtype=int).ravel()

    x_sub = np.log1p(np.maximum(x_sub, 0))
    return x_sub, y_sub


def prepare_tmdb_reg(
    X: pd.DataFrame,
    y: pd.Series,
    feats: list[str] | None = None,
    *,
    max_rows: int = 2000,
    random_state: int = 0,
    log_y: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    feats = resolve_features(X, feats)
    if len(X) > max_rows:
        rng = np.random.default_rng(random_state)
        idx = rng.choice(len(X), size=max_rows, replace=False)
        x_sub = X.iloc[idx][feats].values.astype(float)
        y_sub = np.asarray(y, dtype=float).ravel()[idx]
    else:
        x_sub = X[feats].values.astype(float)
        y_sub = np.asarray(y, dtype=float).ravel()

    x_sub = np.log1p(np.maximum(x_sub, 0))
    if log_y:
        y_sub = np.log1p(np.maximum(y_sub, 0))
    return x_sub, y_sub


def prepare_tmdb_features(
    X: pd.DataFrame,
    y: pd.Series,
    feats: list[str] | None = None,
    *,
    max_rows: int = 3000,
    random_state: int = 0,
    log_transform: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    feats = resolve_features(X, feats)
    X, y = subsample_xy(X, y, max_rows=max_rows, random_state=random_state)

    x_arr = X[feats].values.astype(float)
    y_arr = np.asarray(y, dtype=float)
    if log_transform:
        x_arr = np.log1p(np.maximum(x_arr, 0))
        y_arr = np.log1p(np.maximum(y_arr, 0))
    return x_arr, y_arr


def prepare_tmdb_unsupervised(
    X: pd.DataFrame,
    feats: list[str] | None = None,
    *,
    max_rows: int = 2000,
    random_state: int = 0,
) -> np.ndarray:
    feats = resolve_features(X, feats)
    if len(X) > max_rows:
        rng = np.random.default_rng(random_state)
        idx = rng.choice(len(X), size=max_rows, replace=False)
        x_sub = X.iloc[idx][feats].values.astype(float)
    else:
        x_sub = X[feats].values.astype(float)

    return np.log1p(np.maximum(x_sub, 0))


def log_feature_xy(
    X: pd.DataFrame,
    y: pd.Series,
    feat: str,
    *,
    max_rows: int = 3000,
    random_state: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """Subsample and log1p-transform a single feature and the target."""
    X, y = subsample_xy(X, y, max_rows=max_rows, random_state=random_state)
    x = np.log1p(X[feat].values.astype(float)).reshape(-1, 1)
    y_log = np.log1p(np.asarray(y, dtype=float))
    return x, y_log


def log_features_xy(
    X: pd.DataFrame,
    y: pd.Series,
    feats: list[str],
    *,
    max_rows: int = 3000,
    random_state: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    X, y = subsample_xy(X, y, max_rows=max_rows, random_state=random_state)
    cols = [np.log1p(X[f].values.astype(float)) for f in feats]
    x_arr = np.column_stack(cols)
    y_log = np.log1p(np.asarray(y, dtype=float))
    return x_arr, y_log


def single_feature_subsample(
    X: pd.DataFrame,
    y: pd.Series,
    feat: str,
    *,
    max_rows: int = 3000,
    random_state: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """Return sorted (x, y) arrays for a single feature, subsampled for speed."""
    X, y = subsample_xy(X, y, max_rows=max_rows, random_state=random_state)
    x = X[feat].values.astype(float)
    order = np.argsort(x)
    return x[order], np.asarray(y)[order]


def subsample_log1p_xy(
    X: pd.DataFrame,
    y: pd.Series,
    feat: str,
    *,
    max_rows: int = 2000,
    random_state: int = 0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Subsample, sort, and log1p-transform (x, y) for a single feature."""
    X, y = subsample_xy(X, y, max_rows=max_rows, random_state=random_state)
    x_raw = X[feat].values.astype(float)
    y_raw = np.asarray(y, dtype=float)
    order = np.argsort(x_raw)
    x_raw, y_raw = x_raw[order], y_raw[order]
    return x_raw, y_raw, np.log1p(x_raw), np.log1p(y_raw)


def prepare_tmdb_numeric_matrix(
    x_df: pd.DataFrame,
    feats: list[str] | None = None,
    *,
    max_rows: int = 1000,
    random_state: int = 0,
) -> tuple[np.ndarray, list[str]]:
    feats = resolve_features(x_df, feats)
    x_df = x_df[feats].select_dtypes(include=[np.number]).fillna(0.0)

    if len(x_df) > max_rows:
        rng = np.random.default_rng(random_state)
        idx = rng.choice(len(x_df), size=max_rows, replace=False)
        x_df = x_df.iloc[idx]

    x_raw = np.log1p(np.maximum(x_df.values.astype(float), 0.0))
    return x_raw, feats


def prepare_tmdb_all_numeric_log1p(
    x_df: pd.DataFrame,
    y: pd.Series,
    *,
    max_rows: int = 600,
    random_state: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    x_df, y = subsample_xy(x_df, y, max_rows=max_rows, random_state=random_state)
    x_num = x_df.select_dtypes(include=[np.number]).fillna(0.0)
    x_base = np.log1p(np.maximum(x_num.values.astype(float), 0.0))
    y_log = np.log1p(np.maximum(np.asarray(y, dtype=float), 0.0))
    return x_base, y_log
