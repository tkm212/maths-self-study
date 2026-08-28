"""Quantitative finance utilities (López de Prado, AFML).

Pipeline: bars → CUSUM filter → triple-barrier labels → sample weights.
"""

from maths_self_study.quant.bars import dollar_bars, tick_bars, time_bars, volume_bars
from maths_self_study.quant.filters import cusum_filter
from maths_self_study.quant.labeling import triple_barrier_labels
from maths_self_study.quant.weights import (
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
