"""Run and bootstrap utilities for textbook chapter Dash dashboards."""

from __future__ import annotations

import importlib
import sys
from pathlib import Path

from dash import Dash

from maths_self_study.dashboards.chapter_app import DashboardPage
from maths_self_study.dashboards.logging import configure_for_run, log_dashboard_start


def setup_chapter_path(chapter_dir: Path) -> None:
    """Insert a chapter directory on ``sys.path`` for local page and helper imports.

    Chapter folders use short module names such as ``ch3_pages`` and ``ch3_helpers``.
    Those names repeat across textbooks (ESL, DL, AFML), so only the active chapter
    directory may be on ``sys.path`` at once. Package code under ``maths_self_study``
    does not need this bootstrap.
    """
    path = str(chapter_dir.resolve())
    if path not in sys.path:
        sys.path.insert(0, path)


def load_chapter_pages(
    dashboard_file: str | Path,
    pages_module: str,
    *page_names: str,
) -> list[DashboardPage]:
    """Import tab page classes from a chapter-local ``*_pages`` package."""
    setup_chapter_path(Path(dashboard_file).resolve().parent)
    module = importlib.import_module(pages_module)
    return [getattr(module, name) for name in page_names]


def run_dashboard(app: Dash, *, debug: bool = True) -> None:
    """Start a chapter dashboard dev server."""
    app.run(debug=debug)


def main_dashboard(app: Dash, *, label: str, debug: bool = True) -> None:
    """Configure logging and run a chapter dashboard."""
    configure_for_run(debug=debug)
    log_dashboard_start(label, debug=debug)
    run_dashboard(app, debug=debug)
