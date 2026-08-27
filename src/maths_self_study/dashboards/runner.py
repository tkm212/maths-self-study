"""Run and bootstrap utilities for textbook chapter Dash dashboards."""

from __future__ import annotations

import sys
from pathlib import Path

from dash import Dash

from maths_self_study.dashboards.logging import configure_for_run, log_dashboard_start


def setup_chapter_path(chapter_dir: Path) -> None:
    """Insert a chapter directory on ``sys.path`` for local ``ch{N}_pages`` imports."""
    path = str(chapter_dir.resolve())
    if path not in sys.path:
        sys.path.insert(0, path)


def run_dashboard(app: Dash, *, debug: bool = True) -> None:
    """Start a chapter dashboard dev server."""
    app.run(debug=debug)


def main_dashboard(app: Dash, *, label: str, debug: bool = True) -> None:
    """Configure logging and run a chapter dashboard."""
    configure_for_run(debug=debug)
    log_dashboard_start(label, debug=debug)
    run_dashboard(app, debug=debug)
