"""TMDB movie metadata dataset loaders."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

TMDB_SUBDIR = "tmdb-movie-metadata"


def load_tmdb_revenue_regression(inputs_dir: Path) -> tuple[pd.DataFrame, pd.Series, str]:
    """
    Load TMDB movie CSV(s) from ``inputs_dir / tmdb-movie-metadata`` (see
    ``scripts/download_tmdb_movie_metadata.py``). Regress **revenue** on numeric
    features: ``budget``, ``popularity``, ``runtime``, ``vote_average``, ``vote_count``,
    and **release_year** parsed from ``release_date`` when present.

    Rows with missing or non-positive revenue/budget are dropped so the target is
    usable for log-scale or standard regression.
    """
    root = inputs_dir / TMDB_SUBDIR
    if not root.is_dir():
        msg = f"Missing {root}. Run: uv run python scripts/download_tmdb_movie_metadata.py"
        raise FileNotFoundError(msg)

    candidates = sorted(root.rglob("*.csv"))
    preferred = [p for p in candidates if p.name.lower() == "tmdb_5000_movies.csv"]
    if not preferred:
        preferred = [p for p in candidates if "movie" in p.name.lower() and "credit" not in p.name.lower()]
    if not preferred:
        preferred = candidates
    if not preferred:
        msg = f"No CSV files under {root}"
        raise FileNotFoundError(msg)

    path = preferred[0]
    df = pd.read_csv(path, low_memory=False)

    target = "revenue"
    if target not in df.columns:
        msg = f"Expected column {target!r} in {path.name}; got {list(df.columns)[:25]}"
        raise ValueError(msg)

    df = df.copy()
    for col in ("budget", "revenue"):
        df[col] = pd.to_numeric(df[col], errors="coerce")

    if "release_date" in df.columns:
        df["release_year"] = pd.to_datetime(df["release_date"], errors="coerce").dt.year

    df = df[df["revenue"] > 0]
    df = df[df["budget"] > 0]

    y_raw = df[target]
    drop = {target, "id", "imdb_id"}
    X = df.drop(columns=[c for c in drop if c in df.columns], errors="ignore")
    X = X.select_dtypes(include=[np.number])

    tbl = pd.concat([y_raw, X], axis=1).dropna()
    y = tbl[target]
    X = tbl.drop(columns=[target])

    if len(X) < 50:
        msg = f"Too few complete rows after filters: {len(X)}"
        raise ValueError(msg)

    return X, y, target


def load_tmdb_revenue_classification(inputs_dir: Path) -> tuple[pd.DataFrame, pd.Series, str]:
    """
    Binary classification on TMDB movies: ``high_revenue = 1`` if revenue >= median, else 0.

    Uses the same feature set as :func:`load_tmdb_revenue_regression` so all Chapter 3 helpers
    that call ``load_tmdb_xy`` continue to work when swapped for this loader.
    """
    X, y_reg, _ = load_tmdb_revenue_regression(inputs_dir)
    target = "high_revenue"
    y = (y_reg >= y_reg.median()).astype(int)
    y.name = target
    return X, y, target
