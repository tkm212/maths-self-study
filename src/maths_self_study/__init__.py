"""Maths self-study utilities.

Shared code for textbook-driven notebooks in statistics, machine learning, and
quantitative finance.

Public API: bars, filters, labeling, and sample weights from López de Prado (2018).

``loaders`` is intentionally not re-exported here — it requires external
datasets (downloaded via Kaggle) and is only used by textbook notebooks.
Import it directly: ``from maths_self_study.loaders import ...``
"""

from maths_self_study.bars import dollar_bars, tick_bars, time_bars, volume_bars
from maths_self_study.filters import cusum_filter
from maths_self_study.labeling import triple_barrier_labels
from maths_self_study.weights import (
    average_uniqueness,
    concurrent_labels_per_bar,
    time_decay_weights,
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
