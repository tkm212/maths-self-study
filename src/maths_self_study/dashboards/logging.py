"""Logging helpers for chapter Dash dashboards."""

from __future__ import annotations

import logging

LOGGER = logging.getLogger("maths_self_study.dashboards")


def configure(*, level: int = logging.INFO, force: bool = False) -> None:
    """Configure concise dashboard logging (idempotent unless force=True)."""
    logging.basicConfig(
        level=level,
        format="%(levelname)s %(name)s: %(message)s",
        force=force,
    )
