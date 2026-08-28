"""Cached data loaders for AFML chapter dashboards."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import pandas as pd

from maths_self_study.data.notebooks import init_paths

TICKS_FILENAME = "btc_bid_ask_data.csv"
TIME_BARS_FILENAME = "btc_bid_ask_data_1s.csv"


class DataNotFoundError(FileNotFoundError):
    """Raised when required AFML demo data is missing from inputs/ or outputs/."""


def _require(path: Path, *, hint: str) -> Path:
    if not path.exists():
        msg = f"Missing data file: {path}\n{hint}"
        raise DataNotFoundError(msg)
    return path


@lru_cache(maxsize=1)
def project_paths() -> tuple[Path, Path, Path]:
    return init_paths()


@lru_cache(maxsize=1)
def load_ticks() -> pd.DataFrame:
    """Load and clean BTC tick data from ``inputs/btc_bid_ask_data.csv``."""
    _, inputs, _ = project_paths()
    path = _require(
        inputs / TICKS_FILENAME,
        hint="Place tick CSV under inputs/ or run scripts/generate_all_bars.py after obtaining data.",
    )
    df = pd.read_csv(path)
    df = df.dropna(subset=["time", "Price", "Quantity"])
    df = df[(df["time"] >= 1e9) & (df["Price"] > 0) & (df["Quantity"] > 0)]
    return df.sort_values("time").reset_index(drop=True)


@lru_cache(maxsize=1)
def load_time_bars() -> pd.DataFrame:
    """Load 1-second bars from ``outputs/btc_bid_ask_data_1s.csv``."""
    _, _, outputs = project_paths()
    path = _require(
        outputs / TIME_BARS_FILENAME,
        hint="Run the Bar Types page first or execute scripts/generate_all_bars.py.",
    )
    bars = pd.read_csv(path)
    bars["datetime"] = pd.to_datetime(bars["datetime"])
    return bars
