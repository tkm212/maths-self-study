"""Tests for TMDB feature preparation helpers."""

from __future__ import annotations

import numpy as np
import pandas as pd

from maths_self_study.data.tmdb_features import (
    log_feature_xy,
    prepare_tmdb_all_numeric_log1p,
    prepare_tmdb_cls,
    prepare_tmdb_features,
    prepare_tmdb_numeric_matrix,
    prepare_tmdb_reg,
    prepare_tmdb_unsupervised,
    single_feature_subsample,
    subsample_log1p_xy,
)


def _sample_frame(n: int = 100) -> tuple[pd.DataFrame, pd.Series]:
    rng = np.random.default_rng(0)
    X = pd.DataFrame(
        {
            "budget": rng.integers(1, 1000, size=n),
            "popularity": rng.random(n),
            "runtime": rng.integers(60, 180, size=n),
            "vote_average": rng.random(n) * 10,
            "vote_count": rng.integers(1, 5000, size=n),
        }
    )
    y_reg = pd.Series(rng.integers(1, 5000, size=n))
    return X, y_reg


def test_prepare_tmdb_cls_shapes() -> None:
    X, y = _sample_frame()
    y_cls = (y >= y.median()).astype(int)
    x_arr, y_arr = prepare_tmdb_cls(X, y_cls, max_rows=50, random_state=0)
    assert x_arr.shape == (50, 5)
    assert y_arr.shape == (50,)
    assert set(np.unique(y_arr)).issubset({0, 1})


def test_prepare_tmdb_reg_log_y() -> None:
    X, y = _sample_frame()
    x_arr, y_arr = prepare_tmdb_reg(X, y, max_rows=40, random_state=0)
    assert x_arr.shape == (40, 5)
    assert y_arr.shape == (40,)
    assert np.all(y_arr >= 0)


def test_prepare_tmdb_features_toggle_log() -> None:
    X, y = _sample_frame()
    x_raw, y_raw = prepare_tmdb_features(X, y, max_rows=30, log_transform=False)
    assert x_raw.shape == (30, 5)
    assert np.any(x_raw > 10)


def test_log_feature_xy_single_column() -> None:
    X, y = _sample_frame()
    x, y_log = log_feature_xy(X, y, "budget", max_rows=25)
    assert x.shape == (25, 1)
    assert y_log.shape == (25,)


def test_single_feature_subsample_sorted() -> None:
    X, y = _sample_frame()
    x, _y = single_feature_subsample(X, y, "budget", max_rows=20)
    assert np.all(np.diff(x) >= 0)


def test_subsample_log1p_xy_sorted() -> None:
    X, y = _sample_frame()
    x_raw, _y_raw, x_fit, y_fit = subsample_log1p_xy(X, y, "budget", max_rows=20)
    assert np.all(np.diff(x_raw) >= 0)
    assert x_fit.shape == x_raw.shape
    assert y_fit.shape == _y_raw.shape


def test_prepare_tmdb_unsupervised() -> None:
    X, _y = _sample_frame()
    arr = prepare_tmdb_unsupervised(X, max_rows=35)
    assert arr.shape == (35, 5)


def test_prepare_tmdb_numeric_matrix() -> None:
    X, _y = _sample_frame()
    mat, feats = prepare_tmdb_numeric_matrix(X, max_rows=30)
    assert mat.shape == (30, len(feats))
    assert feats == ["budget", "popularity", "runtime", "vote_average", "vote_count"]


def test_prepare_tmdb_all_numeric_log1p() -> None:
    X, y = _sample_frame()
    x_base, y_log = prepare_tmdb_all_numeric_log1p(X, y, max_rows=25)
    assert x_base.shape[0] == 25
    assert y_log.shape == (25,)
