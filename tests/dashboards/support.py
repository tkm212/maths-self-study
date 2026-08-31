"""Shared helpers for dashboard tests (not collected by pytest)."""

from __future__ import annotations

import importlib.util
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]

CH2_DASHBOARD = REPO_ROOT / "textbooks/deep-learning/2-Linear_Algebra/dashboard.py"
CH3_DASHBOARD = REPO_ROOT / "textbooks/deep-learning/3-Probability_Information_Theory/dashboard.py"
CH4_DASHBOARD = REPO_ROOT / "textbooks/deep-learning/4-Numerical_Computation/dashboard.py"
CH5_DASHBOARD = REPO_ROOT / "textbooks/deep-learning/5-Machine_Learning_Basics/dashboard.py"
ESL_CH2_DASHBOARD = REPO_ROOT / "textbooks/elements-of-statistical-learning/2-Supervised_Learning/dashboard.py"
ESL_CH4_DASHBOARD = (
    REPO_ROOT / "textbooks/elements-of-statistical-learning/4-Linear_Methods_Classification/dashboard.py"
)
ESL_CH7_DASHBOARD = REPO_ROOT / "textbooks/elements-of-statistical-learning/7-Model_Assessment/dashboard.py"

CHAPTER_MODULE_ROOTS = tuple(f"ch{n}_{suffix}" for n in range(2, 19) for suffix in ("pages", "helpers", "data"))


def clear_chapter_modules() -> None:
    import sys

    for name in list(sys.modules):
        if any(name == root or name.startswith(f"{root}.") for root in CHAPTER_MODULE_ROOTS):
            del sys.modules[name]


def prepare_chapter_import(chapter_dir: Path) -> None:
    import sys

    clear_chapter_modules()
    chapter_dir_str = str(chapter_dir.resolve())
    sys.path[:] = [path for path in sys.path if path != chapter_dir_str]
    sys.path.insert(0, chapter_dir_str)


def load_dashboard_module(path: Path):
    import sys

    clear_chapter_modules()
    chapter_dir = str(path.parent.resolve())
    if chapter_dir in sys.path:
        sys.path.remove(chapter_dir)
    sys.path.insert(0, chapter_dir)

    module_name = "dashboard_" + path.parent.as_posix().replace("/", "_").replace("-", "_")
    spec = importlib.util.spec_from_file_location(module_name, path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module
