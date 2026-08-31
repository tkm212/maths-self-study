"""Cached TMDB data for ESL Ch. 3 dashboard pages."""

from __future__ import annotations

from functools import lru_cache

import ch3_helpers as helpers


@lru_cache(maxsize=1)
def load_scaled():
    _root, inputs, _outputs = helpers.init_paths()
    X, y, target = helpers.load_tmdb_xy(inputs)
    return helpers.scale_split(X, y), target
