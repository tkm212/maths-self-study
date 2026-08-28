"""Tests for textbook notebook bootstrap helpers."""

from __future__ import annotations

import numpy as np
import pandas as pd

from maths_self_study.data.notebooks import find_project_root, scale_split


def test_find_project_root_from_repo() -> None:
    root = find_project_root()
    assert (root / "pyproject.toml").is_file()


def test_scale_split_shapes() -> None:
    rng = np.random.default_rng(0)
    X = pd.DataFrame(rng.normal(size=(40, 3)), columns=["a", "b", "c"])
    y = pd.Series(rng.normal(size=40))
    out = scale_split(X, y, test_size=0.25, random_state=0)
    assert out["X_train"].shape == (30, 3)
    assert out["X_test"].shape == (10, 3)
    assert out["X_train_s"].shape == (30, 3)
    assert out["feat_names"] == ["a", "b", "c"]
