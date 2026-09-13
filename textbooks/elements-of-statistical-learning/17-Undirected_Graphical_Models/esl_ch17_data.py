"""Cached paths for ESL Ch. 17 dashboard pages."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import esl_ch17_helpers as helpers


@lru_cache(maxsize=1)
def load_inputs() -> Path:
    _root, inputs, _ = helpers.init_paths()
    return inputs
