"""Maths self-study utilities.

Shared code for textbook-driven notebooks in statistics, machine learning, and
quantitative finance.

Public API: quant pipeline (bars, filters, labeling, weights) from López de Prado (2018).

``data`` loaders are intentionally not re-exported here — they require external
datasets (downloaded via Kaggle) and are only used by textbook notebooks.
Import them directly: ``from maths_self_study.data import ...``
"""

from maths_self_study.quant import (
    average_uniqueness,
    concurrent_labels_per_bar,
    cusum_filter,
    dollar_bars,
    tick_bars,
    time_bars,
    time_decay_weights,
    triple_barrier_labels,
    volume_bars,
)

__all__ = [
    "average_uniqueness",
    "concurrent_labels_per_bar",
    "cusum_filter",
    "dollar_bars",
    "tick_bars",
    "time_bars",
    "time_decay_weights",
    "triple_barrier_labels",
    "volume_bars",
]
