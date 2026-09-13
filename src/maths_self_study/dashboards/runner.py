"""Run and bootstrap utilities for textbook chapter Dash dashboards."""

from __future__ import annotations

import importlib
import sys
from functools import lru_cache
from pathlib import Path

from dash import Dash

from maths_self_study.dashboards.chapter_app import DashboardPage
from maths_self_study.dashboards.logging import configure_for_run, log_dashboard_start
from maths_self_study.data.notebooks import find_project_root

_PATHS_ENSURED = False


@lru_cache(maxsize=1)
def textbook_chapter_dirs(root: Path | None = None) -> tuple[Path, ...]:
    """Return every chapter folder that contains a ``dashboard.py``."""
    base = root or find_project_root()
    return tuple(sorted({path.parent.resolve() for path in (base / "textbooks").glob("**/dashboard.py")}))


def ensure_textbook_chapter_paths(root: Path | None = None) -> None:
    """Register all textbook chapter folders on ``sys.path`` once.

    Chapter packages use book-unique names such as ``esl_ch03_pages`` and
    ``dl_ch03_pages``, so every chapter directory can safely coexist on the path.
    """
    global _PATHS_ENSURED
    if _PATHS_ENSURED:
        return
    for chapter_dir in textbook_chapter_dirs(root):
        path = str(chapter_dir)
        if path not in sys.path:
            sys.path.append(path)
    _PATHS_ENSURED = True


def setup_chapter_path(chapter_dir: Path) -> None:
    """Ensure textbook chapter paths are registered (legacy alias)."""
    ensure_textbook_chapter_paths()
    path = str(chapter_dir.resolve())
    if path not in sys.path:
        sys.path.append(path)


def load_chapter_pages(
    dashboard_file: str | Path,
    pages_module: str,
    *page_names: str,
) -> list[DashboardPage]:
    """Import tab page classes from a book-unique chapter pages package."""
    ensure_textbook_chapter_paths()
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
