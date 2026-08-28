"""Textbook dataset loaders (requires data under ``inputs/``)."""

from maths_self_study.data.atpwta import load_atpwta_regression
from maths_self_study.data.notebooks import init_paths
from maths_self_study.data.tmdb import (
    load_tmdb_revenue_classification,
    load_tmdb_revenue_regression,
)

__all__ = [
    "init_paths",
    "load_atpwta_regression",
    "load_tmdb_revenue_classification",
    "load_tmdb_revenue_regression",
]
