"""Cached TMDB data for ESL Ch. 14 dashboard pages."""

from __future__ import annotations

from functools import lru_cache

import ch14_helpers as helpers


@lru_cache(maxsize=1)
def load_x():
    _root, inputs, _ = helpers.init_paths()
    return helpers.load_tmdb_xy(inputs)
