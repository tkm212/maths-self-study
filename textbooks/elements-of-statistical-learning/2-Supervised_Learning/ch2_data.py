"""Cached TMDB data for ESL Ch. 2 dashboard pages."""

from __future__ import annotations

from functools import lru_cache

import helpers


@lru_cache(maxsize=1)
def load_xy():
    _root, inputs, _outputs = helpers.init_paths()
    return helpers.load_tmdb_xy(inputs)
