"""ATP/WTA tennis match dataset loader."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

ATPWTA_SUBDIR = "atpwta-tennis-data"

TARGET_PRIORITY = (
    "winner_rank_points",
    "minutes",
    "loser_rank_points",
)


def load_atpwta_regression(inputs_dir: Path) -> tuple[pd.DataFrame, pd.Series, str]:
    """
    Read ATP/WTA *match* CSVs from ``inputs_dir / atpwta-tennis-data`` (see
    ``scripts/download_atpwta_tennis_data.py``) and return numeric ``X``, continuous ``y``.

    Uses the first available target in order: ``winner_rank_points``, ``minutes``,
    ``loser_rank_points`` (common in Sackmann-style match files).
    """
    root = inputs_dir / ATPWTA_SUBDIR
    if not root.is_dir():
        msg = f"Missing {root}. Run: uv run python scripts/download_atpwta_tennis_data.py"
        raise FileNotFoundError(msg)

    paths = sorted(root.rglob("*.csv"))
    match_paths = [p for p in paths if "match" in p.name.lower()]
    use_paths = match_paths or paths
    if not use_paths:
        msg = f"No CSV files under {root}"
        raise FileNotFoundError(msg)

    frames: list[pd.DataFrame] = []
    for p in use_paths:
        try:
            frames.append(pd.read_csv(p, low_memory=False))
        except OSError, UnicodeDecodeError, pd.errors.EmptyDataError:
            continue

    if not frames:
        msg = f"Could not read any CSV from {root}"
        raise ValueError(msg)

    df = pd.concat(frames, ignore_index=True)

    target = next((c for c in TARGET_PRIORITY if c in df.columns), None)
    if target is None:
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        msg = f"No target in {TARGET_PRIORITY}. Numeric columns (sample): {numeric_cols[:20]}"
        raise ValueError(msg)

    y_raw = df[target]
    X = df.drop(columns=[target], errors="ignore").select_dtypes(include=[np.number])
    tbl = pd.concat([y_raw, X], axis=1).dropna()
    y = tbl[target]
    X = tbl.drop(columns=[target])

    if len(X) < 50:
        msg = f"Too few complete rows after dropna: {len(X)}"
        raise ValueError(msg)

    return X, y, target
