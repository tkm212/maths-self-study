"""Shared notebook bootstrap helpers for textbook marimo notebooks."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from maths_self_study.data.tmdb import load_tmdb_revenue_classification, load_tmdb_revenue_regression

__all__ = [
    "ensure_package_on_path",
    "find_project_root",
    "init_paths",
    "load_tmdb_classification_xy",
    "load_tmdb_cls",
    "load_tmdb_regression_xy",
    "load_tmdb_revenue_xy",
    "load_tmdb_xy",
    "scale_split",
]


def find_project_root(max_up: int = 12) -> Path:
    p = Path.cwd().resolve()
    for _ in range(max_up):
        if (p / "pyproject.toml").exists():
            return p
        if p.parent == p:
            break
        p = p.parent
    msg = "Could not find project root (pyproject.toml)."
    raise RuntimeError(msg)


def ensure_package_on_path(root: Path) -> None:
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))


def init_paths() -> tuple[Path, Path, Path]:
    """Return ``(project_root, inputs_dir, outputs_dir)`` and put the package on ``sys.path``."""
    root = find_project_root()
    ensure_package_on_path(root)
    return root, root / "inputs", root / "outputs"


def load_tmdb_xy(inputs_dir: Path) -> tuple[pd.DataFrame, pd.Series, str]:
    return load_tmdb_revenue_regression(inputs_dir)


load_tmdb_regression_xy = load_tmdb_xy
load_tmdb_revenue_xy = load_tmdb_xy


def load_tmdb_classification_xy(inputs_dir: Path) -> tuple[pd.DataFrame, pd.Series, str]:
    return load_tmdb_revenue_classification(inputs_dir)


def load_tmdb_cls(inputs_dir: Path) -> tuple[pd.DataFrame, pd.Series, str]:
    """Binary classification: high_revenue = 1 if revenue >= median (else 0)."""
    return load_tmdb_revenue_classification(inputs_dir)


def scale_split(
    X: pd.DataFrame,
    y: pd.Series,
    *,
    test_size: float = 0.25,
    random_state: int = 0,
) -> dict[str, Any]:
    """Train/test split then standard-scale X. Returns a dict with all pieces."""
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)
    return {
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
        "X_train_s": X_train_s,
        "X_test_s": X_test_s,
        "scaler": scaler,
        "feat_names": list(X_train.columns),
    }
