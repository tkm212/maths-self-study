"""Cached TMDB data for ESL Ch. 16 dashboard pages."""

from __future__ import annotations

from functools import lru_cache

import ch16_helpers as helpers


@lru_cache(maxsize=1)
def load_xy():
    _root, inputs, _ = helpers.init_paths()
    return helpers.load_tmdb_classification_xy(inputs)


@lru_cache(maxsize=1)
def load_regression_xy():
    _root, inputs, _ = helpers.init_paths()
    return helpers.load_tmdb_regression_xy(inputs)
