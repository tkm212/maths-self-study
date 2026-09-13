"""Tests for AFML BTC data loaders."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import pandas as pd
import pytest

from maths_self_study.data import fml


@pytest.fixture(autouse=True)
def _clear_fml_caches() -> Iterator[None]:
    fml.project_paths.cache_clear()
    fml.load_ticks.cache_clear()
    fml.load_time_bars.cache_clear()
    yield
    fml.project_paths.cache_clear()
    fml.load_ticks.cache_clear()
    fml.load_time_bars.cache_clear()


def test_load_ticks_cleans_and_sorts(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    inputs = tmp_path / "inputs"
    outputs = tmp_path / "outputs"
    inputs.mkdir()
    outputs.mkdir()
    monkeypatch.setattr(fml, "project_paths", lambda: (tmp_path, inputs, outputs))

    csv = inputs / fml.TICKS_FILENAME
    pd.DataFrame({
        "time": [1e9 - 1.0, 1e9, 1e9 + 1.0, 1e9 + 2.0],
        "Price": [0.0, 100.0, 101.0, 102.0],
        "Quantity": [0.0, 1.0, 2.0, 1.5],
    }).to_csv(csv, index=False)

    ticks = fml.load_ticks()
    assert len(ticks) == 3
    assert ticks["time"].is_monotonic_increasing
    assert (ticks["Price"] > 0).all()


def test_load_ticks_missing_file(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    inputs = tmp_path / "inputs"
    outputs = tmp_path / "outputs"
    inputs.mkdir()
    outputs.mkdir()
    monkeypatch.setattr(fml, "project_paths", lambda: (tmp_path, inputs, outputs))

    with pytest.raises(fml.DataNotFoundError, match="Missing data file"):
        fml.load_ticks()


def test_load_time_bars_parses_datetime(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    inputs = tmp_path / "inputs"
    outputs = tmp_path / "outputs"
    inputs.mkdir()
    outputs.mkdir()
    monkeypatch.setattr(fml, "project_paths", lambda: (tmp_path, inputs, outputs))

    csv = outputs / fml.TIME_BARS_FILENAME
    pd.DataFrame({
        "datetime": ["2024-01-01 09:00:00", "2024-01-01 09:00:01"],
        "open": [100.0, 100.1],
        "high": [100.2, 100.3],
        "low": [99.9, 100.0],
        "close": [100.1, 100.2],
    }).to_csv(csv, index=False)

    bars = fml.load_time_bars()
    assert pd.api.types.is_datetime64_any_dtype(bars["datetime"])
    assert len(bars) == 2
