"""Cached TMDB data for ESL Ch. 9 dashboard pages."""

from __future__ import annotations

from functools import lru_cache

import ch9_helpers as helpers


@lru_cache(maxsize=1)
def load_xy():
    _root, inputs, _outputs = helpers.init_paths()
    return helpers.load_tmdb_xy(inputs)
