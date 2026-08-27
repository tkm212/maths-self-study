"""Backward-compatible re-exports — use ``maths_self_study.demos.deep_learning`` instead."""

from maths_self_study.demos.deep_learning import ch2 as ch2_helpers
from maths_self_study.demos.deep_learning import ch3 as ch3_helpers
from maths_self_study.demos.deep_learning import ch4 as ch4_helpers
from maths_self_study.demos.deep_learning import ch5 as ch5_helpers
from maths_self_study.demos.deep_learning.dashboard import create_deep_learning_dashboard

__all__ = [
    "ch2_helpers",
    "ch3_helpers",
    "ch4_helpers",
    "ch5_helpers",
    "create_deep_learning_dashboard",
]
