"""Unit tests for maths_self_study.quant.weights."""

import pandas as pd
import pytest

from maths_self_study.quant.weights import (
    average_uniqueness,
    concurrent_labels_per_bar,
    sample_weights_from_bars,
    time_decay_weights,
)


def test_concurrent_labels_per_bar() -> None:
    ev = pd.DataFrame({
        "datetime": [
            pd.Timestamp("2024-01-01 08:00"),
            pd.Timestamp("2024-01-01 09:00"),
            pd.Timestamp("2024-01-01 10:00"),
        ],
        "exit_time": [
            pd.Timestamp("2024-01-01 12:00"),
            pd.Timestamp("2024-01-01 12:00"),
            pd.Timestamp("2024-01-01 12:00"),
        ],
    })
    bars = pd.date_range("2024-01-01 09:00", "2024-01-01 13:00", freq="h")
    c = concurrent_labels_per_bar(ev, bars)
    assert int(c.loc[pd.Timestamp("2024-01-01 10:00")]) == 3
    assert int(c.loc[pd.Timestamp("2024-01-01 12:00")]) == 3


def test_average_uniqueness_single_event() -> None:
    ev = pd.DataFrame({
        "datetime": [pd.Timestamp("2024-01-01 10:00")],
        "exit_time": [pd.Timestamp("2024-01-01 12:00")],
    })
    bars = pd.date_range("2024-01-01 09:00", "2024-01-01 13:00", freq="h")
    u = average_uniqueness(ev, bars)
    assert len(u) == 1
    assert u.iloc[0] == pytest.approx(1.0)


def test_average_uniqueness_full_overlap() -> None:
    ev = pd.DataFrame({
        "datetime": [pd.Timestamp("2024-01-01 10:00")] * 2,
        "exit_time": [pd.Timestamp("2024-01-01 12:00")] * 2,
    })
    bars = pd.date_range("2024-01-01 09:00", "2024-01-01 13:00", freq="h")
    u = average_uniqueness(ev, bars)
    assert u.iloc[0] == pytest.approx(0.5)
    assert u.iloc[1] == pytest.approx(0.5)


def test_time_decay_weights_order() -> None:
    t = pd.Series(pd.date_range("2024-01-01", periods=3, freq="h", tz="UTC"))
    ref = pd.Timestamp("2024-01-02", tz="UTC")
    w = time_decay_weights(t, ref_time=ref, decay_span=pd.Timedelta(hours=6))
    assert w.iloc[2] > w.iloc[1] > w.iloc[0]


def test_time_decay_weights_normalized_span() -> None:
    t = pd.Series([pd.Timestamp("2024-01-01 12:00", tz="UTC")])
    ref = pd.Timestamp("2024-01-01 12:00", tz="UTC")
    w = time_decay_weights(t, ref_time=ref, decay_span=pd.Timedelta(days=1))
    assert w.iloc[0] == pytest.approx(1.0)


def test_sample_weights_from_bars_columns() -> None:
    bars = pd.DataFrame({
        "datetime": pd.date_range("2024-01-01 09:00", periods=30, freq="1min"),
        "open": [100.0] * 30,
        "high": [100.5 + (i % 3) * 0.2 for i in range(30)],
        "low": [99.5 - (i % 2) * 0.1 for i in range(30)],
        "close": [100.0 + (i % 5) * 0.05 for i in range(30)],
    })
    labels, conc = sample_weights_from_bars(
        bars,
        cusum_threshold=0.0001,
        pt=0.001,
        sl=0.001,
        num_bars=5,
        decay_hours=1.0,
    )
    if labels.empty:
        pytest.skip("CUSUM produced no events on synthetic bars")
    assert "avg_uniqueness" in labels.columns
    assert "time_decay" in labels.columns
    assert "sample_weight" in labels.columns
    assert labels["sample_weight"].mean() == pytest.approx(1.0)
    assert len(conc) >= 1
