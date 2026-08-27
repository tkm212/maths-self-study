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


def configure_for_run(*, debug: bool = True) -> None:
    """Set log level for an interactive dashboard dev server."""
    configure(level=logging.DEBUG if debug else logging.INFO, force=True)


def log_dashboard_start(label: str, *, debug: bool = True) -> None:
    """Log that a chapter dashboard is starting."""
    LOGGER.info("Starting %s dashboard (debug=%s)", label, debug)
